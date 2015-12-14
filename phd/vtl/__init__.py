"""Code for calling various parts of VocalTractLab.

VocalTractLab is a high-quality 3D articulatory speech synthesizer
written by Peter Birkholz (http://www.vocaltractlab.de/).
The Linux API used here was compiled by Max Murakami,
with the help of Alex Priamikov and contributions of Anja Philippsen,
in September, 2014.

The API and accompanying files is in the `src` directory.
The functions in this file should obviate the need to use these files,
but if you want to dig deep, here are explanations of those files.

- ``VocalTractLabApi.so``
  The library. The essential file that contains compiled VTL code.
- ``VocalTractLabApi.h``, ``VocalTractLabApi.def``
  Files describing the API entry points in the complied code.
- ``JD2.speaker``
  Speaker file that VTL uses to define the overall characteristics of
  the vocal tract model. JD2 generates an adult male vocal tract.
``child-1y.speaker``
  A VTL speaker file that generates a one-year old child's vocal tract.
"""

import ctypes
import os
import struct
import sys

import numpy as np
from lxml import etree
from nengo.utils.compat import is_number, is_string, iteritems, range
import soundfile as sf

from ..cache import cache_file
from ..utils import hz2st, rescale

# For finding files in `src`
root = os.path.abspath(os.path.dirname(__file__))


def vtl_path(*components):
    pcomponents = (root, 'src') + components
    return os.path.join(*pcomponents)

# For fixing the headers of wavs produced by VTL
WAVHEADER = ('RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'
             +chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)
             +chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'
             +chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)
             +chr(0x00)+'data')

def repair_wavheader(path):
    if not path.lower().endswith('.wav'):
        raise ValueError("Path '%s' does not appear to be "
                         "a .wav file." % path)
    # Get wav contents
    with open(path, 'r') as fp:
        content = fp.read()
    # Back it up, just in case
    os.rename(path, "%s.bak" % path)
    with open(path, 'w') as fp:
        fp.write(WAVHEADER)
        fp.write(content[68:])
    os.remove("%s.bak" % path)


class VTL(object):
    dll = vtl_path('VocalTractLabApi.so')
    try:
        lib = ctypes.cdll.LoadLibrary(dll)
    except OSError:
        lib = None
    default_speaker = vtl_path('JD2.speaker')
    glottis_model = "Triangular glottis"
    numerical_gestures = ['velic', 'f0', 'lung-pressure']
    numerical_range = {'velic': (0., 0.5),  # in arbitrary units
                       'f0': (20., hz2st(520.)),  # in semi-tones
                       'lung-pressure': (0., 1000.)}  # in Pa

    def __init__(self, speaker=None):
        self.speaker = self.default_speaker if speaker is None else speaker

    @property
    def speaker(self):
        return self._speaker

    @speaker.setter
    def speaker(self, _speaker):
        if self.lib is None:
            self._speaker = _speaker
            return

        # Get VTL info
        self.lib.vtlInitialize(_speaker)
        c_int_ptr = ctypes.c_int * 1  # int*
        asr_ptr = c_int_ptr(0)
        nts_ptr = c_int_ptr(0)
        n_vtp_ptr = c_int_ptr(0)
        n_gp_ptr = c_int_ptr(0)
        self.lib.vtlGetConstants(asr_ptr, nts_ptr, n_vtp_ptr, n_gp_ptr)
        self.audio_samplerate = asr_ptr[0]
        self.n_tube_sections = nts_ptr[0]
        self.n_vocaltract_params = n_vtp_ptr[0]
        self.n_glottis_params = n_gp_ptr[0]

        # Get tract info
        c_n_tract_param_ptr = ctypes.c_double * self.n_vocaltract_params
        self.tract_param_names = ctypes.create_string_buffer(
            self.n_vocaltract_params * 32)
        self.tract_param_min = c_n_tract_param_ptr(0)
        self.tract_param_max = c_n_tract_param_ptr(0)
        self.tract_param_neutral = c_n_tract_param_ptr(0)
        self.lib.vtlGetTractParamInfo(self.tract_param_names,
                                      self.tract_param_min,
                                      self.tract_param_max,
                                      self.tract_param_neutral)
        # Note: these are ctypes objects; use np.asarray on them?

        # Get glottis info
        c_n_glottis_param_ptr = ctypes.c_double * self.n_glottis_params
        self.glottis_param_names = ctypes.create_string_buffer(
            self.n_glottis_params * 32)
        self.glottis_param_min = c_n_glottis_param_ptr(0)
        self.glottis_param_max = c_n_glottis_param_ptr(0)
        self.glottis_param_neutral = c_n_glottis_param_ptr(0)
        self.lib.vtlGetGlottisParamInfo(self.glottis_param_names,
                                        self.glottis_param_min,
                                        self.glottis_param_max,
                                        self.glottis_param_neutral)
        # Note: these are ctypes objects; use np.asarray on them?

        self.lib.vtlClose()
        self._speaker = _speaker

    def tract_params(self, shape):
        if self.lib is None:
            return

        c_ntractparam_ptr = ctypes.c_double * self.n_vocaltract_params
        params = c_ntractparam_ptr(0)
        self.lib.vtlInitialize(self.speaker)
        ret = self.lib.vtlGetTractParams(shape, params)
        if ret != 0:
            raise RuntimeError("Shape '%s' is not in the speaker file" % shape)
        return params

    def synthesize(self, gesfile, wavfile=None, areafile=None):
        if self.lib is None:
            raise RuntimeError("VTL can't run on this OS currently.")

        self.lib.vtlInitialize(self.speaker)
        loadwav = wavfile is None
        if wavfile is None:
            wavfile = cache_file(ext='wav')
        ret = self.lib.vtlGesToWav(self.speaker, gesfile, wavfile, areafile)

        # Handle errors
        if ret == 1:
            reason = "Leading speaker file '%s' failed." % self.speaker
        elif ret == 2:
            reason = "Loading gesture score failed."
        elif ret == 3:
            reason = "Invalid wav file name '%s'." % wavfile
        elif ret == 4:
            reason = "wav file '%s' could not be saved." % wavfile
        elif ret != 0:
            reason = "Error code %d" % ret
        if ret != 0:
            raise RuntimeError("VocalTractLab error: %s" % reason)
        self.lib.vtlClose()

        repair_wavheader(wavfile)
        # As a bit of a hack, we'll zero out the first 5 ms, which often has
        # some kind of weird click.
        audio, fs = sf.read(wavfile)
        audio[:int(round(0.005 * fs))] = 0

        if loadwav:
            os.remove(wavfile)
            return audio, fs
        else:
            sf.write(wavfile, audio, fs)

    def synthesize_direct(self, tract_params, glottis_params, duration_s,
                          framerate_hz, wavfile=None):
        extra_frames = 1000
        n_frames = int(duration_s * framerate_hz)
        assert len(tract_params) / self.n_vocaltract_params == n_frames
        assert len(glottis_params) / self.n_glottis_params == n_frames

        # Prep output
        c_int_ptr = ctypes.c_int * 1  # int*
        c_audio_ptr = ctypes.c_double * int(
            duration_s * self.audio_samplerate + extra_frames)
        audio = c_audio_ptr(0)
        n_audio_samples = c_int_ptr(0)
        c_tubeareas_ptr = ctypes.c_double * int(
            n_frames * self.n_tube_sections)
        tubeareas = c_tubeareas_ptr(0)
        c_tractsequence_ptr = ctypes.c_double * int(
            n_frames * self.n_vocaltract_params)
        tract_params_ptr = c_tractsequence_ptr(*tract_params)
        c_glottissequence_ptr = ctypes.c_double * int(
            n_frames * self.n_glottis_params)
        glottis_params_ptr = c_tractsequence_ptr(*glottis_params)

        # Call VTL
        self.lib.vtlInitialize(self.speaker)
        self.lib.vtlSynthBlock(tract_params_ptr,
                               glottis_params_ptr,
                               tubeareas,
                               ctypes.c_int(n_frames),
                               ctypes.c_double(framerate_hz),
                               audio,
                               n_audio_samples)

        # Process output
        out_audio = np.asarray(audio, dtype=np.float64)
        out_audio = np.int16(out_audio / np.max(np.abs(out_audio)) * 32767)
        self.lib.vtlClose()

        if wavfile is None:
            return out_audio, self.audio_samplerate
        sf.write(wavfile, out_audio, self.audio_samplerate)

    def gesture_labels(self):
        """Determines labels for all gestures from the speaker file."""
        labels = []

        # Get the vocal tract shapes first
        xml = etree.parse(self.speaker)
        for elem in xml.iterfind(".//vocal_tract_model/shapes/shape"):
            name = elem.get('name')
            if len(name) > 3 and name[-3] == '(' and name[-1] == ')':
                # The gesture is just the part before the brackets
                name = name[:-3]
            if name not in labels:
                labels.append(name)

        # Get the glottal shape gestures
        gpath = ".//glottis_model[@type='%s']/shapes/shape" % self.glottis_model
        for elem in xml.iterfind(gpath):
            name = elem.get('name')
            if name != 'default':  # Default has no associated gesture
                labels.append(name)

        # Add the numerical gestures
        labels.extend(self.numerical_gestures)
        return labels


def parse_ges(ges_path, speaker=None, ignore_f0=True):
    bridge = VTL(speaker)

    xml = etree.parse(ges_path)
    labels = bridge.gesture_labels()
    if ignore_f0:
        labels.remove('f0')
    gs = GestureScore(labels)
    for element in xml.iter():
        attr = element.attrib
        if element.tag == "gesture_sequence":
            seq = GestureSequence(**attr)
            if ignore_f0 and seq.type.startswith('f0'):
                continue
            gs.sequences.append(seq)
        elif element.tag == "gesture":
            gest_attr = {}
            gest_attr['neutral'] = bool(int(attr.pop('neutral')))
            gest_attr['value'] = attr.pop('value')
            if seq.numerical:
                gest_attr['value'] = float(gest_attr['value'])
            gest_attr.update({key: float(val) for key, val in iteritems(attr)})
            gest = Gesture(**gest_attr)
            seq.gestures.append(gest)
    return gs


def parse_txt(txt_path):
    t = []
    order = ['time', 'tubearea', 'tubelength', 'tract', 'glottis']
    other = {key: [] for key in order}

    with open(txt_path, 'r') as fp:
        data = False
        nextline = None
        for line in fp:
            if line.startswith("#data"):
                data = True
                nextline = order[0]
            elif data:
                linedata = [float(ll) for ll in line.split()]
                if nextline == 'time':
                    t.append(linedata)
                else:
                    other[nextline].append(linedata)
                nextline = order[(int(order.index(nextline)) + 1) % len(order)]

    t = np.asarray(t).ravel()
    for key in other:
        other[key] = np.asarray(other[key])
    return t, other


def synthesize(gesfile, wavfile=None, areafile=None, speaker=None):
    return VTL(speaker).synthesize(gesfile, wavfile, areafile)


class GestureScore(object):
    def __init__(self, labels, sequences=None):
        self.labels = labels
        self.sequences = [] if sequences is None else sequences

    def __xml__(self):
        return etree.tostring(self.__etree__(), pretty_print=True)

    def __etree__(self):
        elem = etree.Element("gestural_score")
        for sequence in self.sequences:
            elem.append(sequence.__etree__())
        return elem

    @property
    def t_end(self):
        if len(self.sequences) == 0:
            return 0.0
        return max(seq.t_end for seq in self.sequences)

    def save(self, path):
        with open(path, 'w') as fp:
            fp.write(self.__xml__())

    def synthesize(self, wavfile=None, areafile=None, speaker=None):
        tmpfile = cache_file(ext='ges')
        self.save(tmpfile)
        ret = VTL(speaker).synthesize(tmpfile, wavfile, areafile)
        os.remove(tmpfile)
        return ret

    def trajectory(self, dt):
        n_steps = int(self.t_end / dt)
        out = np.zeros((n_steps, len(self.labels)))
        for seq in self.sequences:
            out += seq.trajectory(self.t_end, dt, self.labels)
        return out


class GestureSequence(object):
    def __init__(self, type, unit="", gestures=None):
        self.type = type
        self.unit = unit
        self.gestures = [] if gestures is None else gestures

    def __etree__(self):
        elem = etree.Element(
            'gesture_sequence', type=self.type, unit=self.unit)
        for gest in self.gestures:
            elem.append(gest.__etree__())
        return elem

    @property
    def numerical(self):
        return any(self.type.startswith(s) for s in VTL.numerical_gestures)

    @property
    def t_end(self):
        return sum(g.duration_s for g in self.gestures)

    def trajectory(self, t_end, dt, labels):
        """For most gestures, the trajectories are piecewise functions."""
        n_steps = int(t_end / dt)
        out = np.zeros((n_steps, len(labels)))

        if self.numerical:
            label = self.type[:-len("-gestures")]

        t = 0.0
        for gest in self.gestures:
            start = int(t / dt)
            end = start + int(gest.duration_s / dt)
            t += gest.duration_s
            if not self.numerical:
                label = gest.label
            if label == 'neutral':
                continue
            ix = labels.index(label)
            traj = gest.trajectory(dt)
            if self.numerical:
                old_min, old_max = VTL.numerical_range[label]
                traj = rescale(traj, old_min, old_max, 0, 1)
            try:
                out[start:end, ix] = traj[:end-start]
            except ValueError:
                # I have no idea why this happens and is necessary,
                # but it does and it is.
                out[start:end, ix] = traj[:end-start-1]
        return out


class Gesture(object):
    def __init__(self, value, slope, duration_s, time_constant_s, neutral):
        self.value = value
        self.slope = slope  # in st / s
        self.duration_s = duration_s
        self.time_constant_s = time_constant_s
        self.neutral = neutral

    @property
    def label(self):
        if is_number(self.value):
            return 'numerical'
        elif self.neutral or self.value == "":
            return 'neutral'
        return self.value

    @property
    def scalar_value(self):
        if is_number(self.value):
            return self.value
        elif self.neutral or self.value == "":
            return 0.0
        else:
            return 1.0

    def trajectory(self, dt):
        n_steps = int(self.duration_s / dt)
        if self.slope == 0.0:
            return np.ones(n_steps) * self.scalar_value
        val_end = self.scalar_value + (self.slope * self.duration_s)
        return np.linspace(self.scalar_value, val_end, n_steps)

    def __etree__(self):
        return etree.Element('gesture',
                             value=str(self.value),
                             slope=str(self.slope),
                             duration_s=str(self.duration_s),
                             time_constant_s=str(self.time_constant_s),
                             neutral=str(int(self.neutral)))


def get_traindata(gesfile, audio_f, dt,
                  audio_fargs=None, wavfile=None, ignore_f0=True):
    """Get input, output pairs for supervised learning training or testing.

    Parameters
    ----------
    dt : float
        Sampling step size for the gesture and
    gesfile : str
        Path to a .ges gesture file (XML format).
    audio_f : function
        A function that will be applied to the audio stream
    audio_fargs : dict, optional
        Keyword arguments that will be provided to ``audio_f``.
        By default, audio, sampling rate, and dt will be provided.
    wavfile : str, optional
        A .wav file that corresponds to the ``gesfile``.
        If specified but the file does not exist, it will be generated.
        If not specified, audio will be synthesized but not saved.
    """
    gs = parse_ges(gesfile, ignore_f0=ignore_f0)
    y = gs.trajectory(dt=dt)

    if wavfile is None:
        audio, fs = synthesize(gesfile)
    elif not os.path.exists(wavfile):
        synthesize(gesfile, wavfile)
        audio, fs = sf.read(wavfile)
    else:
        audio, fs = sf.read(wavfile)

    audio_fargs = {} if audio_fargs is None else audio_fargs.copy()
    audio_fargs.update({'audio': audio, 'fs': fs, 'dt': dt})
    x = audio_f(**audio_fargs)

    # For some reason, the wav file size and the gesture trajectory size
    # are often off by one or two. Here, we lengthen or shorten ``y``,
    # assuming that VTL is doing it correctly.
    # Not sure if that assumption is correct.
    if x.shape[0] > y.shape[0]:
        # Extend y by n timesteps
        toadd = np.tile(y[np.newaxis, -1], (x.shape[0] - y.shape[0], 1))
        y = np.concatenate((y, toadd))
    if x.shape[0] < y.shape[0]:
        # Shorten y by n timesteps
        todelete = list(range(x.shape[0], y.shape[0]))
        y = np.delete(y, todelete, 0)

    assert x.shape[0] == y.shape[0], "Misaligned; %s %s" % (x.shape, y.shape)
    return x, y, fs
