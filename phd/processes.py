import brian.hears as bh
import nengo
import nengo.utils.numpy as npext
import numpy as np
from scipy.io.wavfile import read as readwav
from scipy.signal import resample

from . import filters


class NengoSound(bh.BaseSound):
    def __init__(self, step_f, nchannels, samplerate):
        self.step_f = step_f
        self.nchannels = nchannels
        self.samplerate = samplerate
        self.t = 0.0
        self.dt = 1. / self.samplerate

    def buffer_init(self):
        pass

    def buffer_fetch(self, start, end):
        return self.buffer_fetch_next(end - start)

    def buffer_fetch_next(self, samples):
        out = np.empty((samples, self.nchannels))
        for i in range(samples):
            self.t += self.dt
            out[i] = self.step_f(self.t)
        return out


class AuditoryFilterBank(nengo.processes.Process):
    def __init__(self, freqs, sound_process, filterbank, samplerate):
        self.freqs = freqs
        self.sound_process = sound_process
        self.filterbank = filterbank
        self.samplerate = samplerate
        super(AuditoryFilterBank, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in[0] == 0
        assert size_out[0] == self.freqs.size

        # If samplerate isn't specified, we'll assume dt
        samplerate = 1. / dt if self.samplerate is None else self.samplerate
        sound_dt = 1. / samplerate

        # Set up the sound
        step_f = self.sound_process.make_step((0,), (1,), sound_dt, rng)
        ns = NengoSound(step_f, 1, samplerate)
        # Always use middle ear filter
        ns = bh.MiddleEar(ns, gain=1)

        # Instantiate filterbank from string
        ihc_cls = getattr(filters, self.filterbank)
        ihc = ihc_cls(ns, self.freqs, dt)
        duration = int(dt / sound_dt)

        def step_filterbank(t, startend=np.array([0, duration], dtype=int)):
            result = ihc.buffer_fetch(startend[0], startend[1])
            startend += duration
            return result[-1]
        return step_filterbank


class FuncProcess(nengo.processes.Process):
    """Psych! Not a process, just a function.

    Implemented so that we can use functions and
    processes interchangeably without having to
    write annoying conditionals.
    """
    def func(self, t):
        raise NotImplementedError()

    def make_step(self, size_in, size_out, dt, rng):
        return self.func


class ArrayProcess(nengo.processes.Process):
    """Psych! Not a process, just going through a vector.

    I guess it's a bit fancier because we can loop
    or stop it after it's done... still though.
    """
    def __init__(self, array, at_end='loop'):
        self.array = array
        # Possible at_end values:
        #   loop: start again from the start
        #   stop: output silence (0) after sound
        assert at_end in ('loop', 'stop')
        self.at_end = at_end
        super(ArrayProcess, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in[0] == 0
        assert size_out[0] == (1 if self.array.ndim == 1
                               else self.array.shape[1])

        rate = 1. / dt

        if self.at_end == 'loop':

            def step_arrayloop(t):
                idx = int(t * rate) % self.array.shape[0]
                return self.array[idx]
            return step_arrayloop

        elif self.at_end == 'stop':

            def step_arraystop(t):
                idx = int(t * rate)
                if idx > self.array.shape[0]:
                    return 0.
                else:
                    return self.array[idx]
            return step_arraystop


class Tone(FuncProcess):
    """A pure tone."""
    def __init__(self, freq_in_hz, rms=0.5):
        self.freq_in_hz = freq_in_hz
        self.rms = rms
        super(Tone, self).__init__()

    @property
    def rms(self):
        return self._rms

    @rms.setter
    def rms(self, _rms):
        self._rms = _rms
        self.amplitude = _rms * np.sqrt(2)

    def func(self, t):
        return self.amplitude * np.sin(2 * np.pi * t * self.freq_in_hz)


class ToneRamp(nengo.processes.Process):
    """Pure tones ramping up over time."""
    def __init__(self, t_ramp=1.0, minfreq=100, maxfreq=8000, rms=0.5):
        self.t_ramp = t_ramp
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.rms = rms
        super(ToneRamp, self).__init__()

    @property
    def rms(self):
        return self._rms

    @rms.setter
    def rms(self, _rms):
        self._rms = _rms
        self.amplitude = _rms * np.sqrt(2)

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in[0] == 0
        assert size_out[0] == 1
        assert dt <= (1. / self.maxfreq)

        n_frames = int(self.t_ramp / dt)
        ramp = np.linspace(self.minfreq, self.maxfreq, n_frames)

        def func(t):
            ix = int(t / dt) % n_frames
            return self.amplitude * np.sin(2 * np.pi * t * ramp[ix])
        return func


class WhiteNoise(nengo.processes.WhiteNoise):
    def __init__(self, rms=0.5):
        # root mean square == standard deviation when mean is 0
        self.rms = rms
        super(WhiteNoise, self).__init__(
            nengo.dists.Gaussian(mean=0, std=rms), scale=False)


class WavFile(nengo.processes.Process):
    def __init__(self, path, at_end='loop', rms=0.5):
        self.path = path
        # Possible at_end values:
        #   loop: start again from the start
        #   stop: output silence (0) after sound
        assert at_end in ('loop', 'stop')
        self.at_end = at_end
        self.rms = rms
        super(WavFile, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in[0] == 0
        assert size_out[0] == 1

        rate = 1. / dt

        orig_rate, orig = readwav(self.path)
        new_size = int(orig.size * (rate / orig_rate))
        wave = resample(orig, new_size)
        wave -= wave.mean()

        # Normalize wave to desired rms
        wave_rms = npext.rms(wave)
        wave *= (self.rms / wave_rms)

        if self.at_end == 'loop':

            def step_wavfileloop(t):
                idx = int(t * rate) % wave.size
                return wave[idx]
            return step_wavfileloop

        elif self.at_end == 'stop':

            def step_wavfilestop(t):
                idx = int(t * rate)
                if idx > wave.size:
                    return 0.
                else:
                    return wave[idx]
            return step_wavfilestop
