import hashlib
import os
import shutil
import tarfile

import dill
import nengo
import nengo.utils.numpy as npext
import numpy as np
import soundfile as sf

from . import config
from .processes import ArrayProcess

consonants = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',
              'jh', 'ch',
              's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',
              'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',
              'l', 'r', 'w', 'y', 'hh', 'hv', 'el']

# closure intervals of stops are distinguished from the stop release

if config.count_closures:
    # Option 1: map the closure to the phoneme
    closures = {'bcl': 'b',
                'dcl': 'd',
                'gcl': 'g',
                'pcl': 'p',
                'tck': 't',
                'kcl': 'k',
                'dcl': 'jh',
                'tcl': 'ch'}
else:
    # Option 2: closure intervals are ignored, considered pauses
    closures = {'bcl': 'pau',
                'dcl': 'pau',
                'gcl': 'pau',
                'pcl': 'pau',
                'tck': 'pau',
                'kcl': 'pau',
                'dcl': 'pau',
                'tcl': 'pau'}

vowels = ['iy', 'ih', 'eh', 'ey',
          'ae', 'aa', 'aw', 'ay', 'ah', 'ao',
          'oy', 'ow', 'uh', 'uw', 'ux',
          'er', 'ax', 'ix', 'axr', 'ax-h']

ignores = ['pau', 'epi', 'h#', '1', '2']

fs = 16000  # TIMIT is always 16 kHz
dt = 1. / fs


def untar(timit_tgz, extract_to):
    # Let's just extract it all and clean up the paths afteward
    with tarfile.open(timit_tgz) as timit_tar:
        timit_tar.extractall(path=extract_to)

    # We don't need the README or the CONVERT or SPHERE directories
    shutil.rmtree(os.path.join(extract_to, 'timit', 'CONVERT'))
    shutil.rmtree(os.path.join(extract_to, 'timit', 'SPHERE'))
    os.remove(os.path.join(extract_to, 'timit', 'README.DOC'))

    # Then, let's move what's left to the root 'timit' directory
    def mvtotimitroot(fileordir):
        shutil.move(os.path.join(extract_to, 'timit', 'TIMIT', fileordir),
                    os.path.join(extract_to, 'timit', fileordir))
    mvtotimitroot('TEST')
    mvtotimitroot('TRAIN')
    mvtotimitroot('README.DOC')

    # Move all docs to the root too
    docdir = os.path.join(extract_to, 'timit', 'TIMIT', 'DOC')
    for doc in os.listdir(docdir):
        shutil.move(os.path.join(docdir, doc),
                    os.path.join(extract_to, 'timit', doc))

    # Remove now-empty TIMIT dir
    shutil.rmtree(os.path.join(extract_to, 'timit', 'TIMIT'))


class Utterance(object):
    def __init__(self, corpus, region, sex, spkr_id, sent_type, sent_number):
        self.corpus = corpus
        self.region = region
        self.sex = sex
        self.spkr_id = spkr_id
        self.sent_type = sent_type
        self.sent_number = sent_number

    @classmethod
    def from_path(cls, path):
        spkr_dir, filename = os.path.split(path)
        filename, _ = os.path.splitext(filename)
        sent_type = filename[1]
        sent_number = int(filename[2:])
        region_dir, spkr_dir = os.path.split(spkr_dir)
        sex = spkr_dir[0]
        spkr_id = spkr_dir[1:]
        corpus_dir, region_dir = os.path.split(region_dir)
        region = int(region_dir[-1])
        _, corpus = os.path.split(corpus_dir)
        return cls(corpus, region, sex, spkr_id, sent_type, sent_number)

    @property
    def path(self):
        return os.path.join(config.timit_root,
                            self.corpus,
                            "DR%d" % self.region,
                            "%s%s" % (self.sex, self.spkr_id),
                            "S%s%d" % (self.sent_type, self.sent_number))

    @property
    def wav(self):
        return "%s.WAV" % self.path

    @property
    def phn(self):
        return "%s.PHN" % self.path


def extract_audio(utterance, phonemes, rms=0.5, t_before=0.):
    """Extract instances of the passed phonemes in the utterance."""
    frames_before = int(t_before / dt)
    ret = {phn: [] for phn in phonemes}
    data, _fs = sf.read(utterance.wav)
    assert _fs == fs, "fs (%s) != 16000" % _fs
    # Normalize data to desired rms
    data_rms = npext.rms(data)
    data *= (rms / data_rms)
    with open(utterance.phn, 'r') as phnfile:
        for line in phnfile:
            start, end, phn = line.split()
            start, end = max(0, int(start) - frames_before), int(end)
            phn = closures[phn] if phn in closures else phn
            if phn not in phonemes:
                continue
            dataslice = np.array(data[start:end])
            ret[phn].append(dataslice)
    return ret


def extract_all_audio(phonemes, rms=0.5, t_before=0.):
    """Generate the audio sequences that will be used."""
    audio = {phn: [] for phn in phonemes}

    def add_to_audio(extracted):
        for phn in audio:
            audio[phn].extend(extracted[phn])
            del extracted[phn]
        del extracted

    corpus_d = os.path.join(config.timit_root, "TRAIN")
    for region_f in os.listdir(corpus_d):
        region_d = os.path.join(corpus_d, region_f)
        for spkr_f in os.listdir(region_d):
            spkr_d = os.path.join(region_d, spkr_f)
            for utt_f in os.listdir(spkr_d):
                if utt_f.endswith('.WAV'):
                    utt_path = os.path.join(spkr_d, utt_f[:-4])
                    utterance = Utterance.from_path(utt_path)
                    add_to_audio(extract_audio(utterance,
                                               phonemes,
                                               rms,
                                               t_before))
    return audio


def ph_traindata(model, detector, sound, target):
    net = model.build(training=True)

    # Put in our own sound and fs in the built network
    net.periphery.sound_process = sound
    net.periphery.fb.sound_process = sound
    net.periphery.fs = fs
    net.periphery.fb.samplerate = fs
    with net:
        # --- Get eval_points
        pr = nengo.Probe(net.detectors[detector.name].phoneme_in,
                         synapse=0.01, sample_every=detector.sample_every)

        # --- Get targets
        # Cache small computations
        t_before = detector.t_before
        zeros = np.zeros_like(target)

        def target_f(t):
            if t > t_before:
                return target
            else:
                return zeros
        target_n = nengo.Node(target_f)
        target_p = nengo.Probe(target_n,
                               synapse=None,
                               sample_every=detector.sample_every)

    sim = nengo.Simulator(net, dt=dt)
    sim.run(dt * sound.array.size, progress_bar=False)
    return sim.data[pr], sim.data[target_p]


def generate_traindata(model, detector, audio):
    """Simulate Sermo with the extracted audio."""
    assert np.allclose(
        detector.sample_every / dt, int(detector.sample_every / dt)), (
            "sample_every must be a multiple of dt (%s)" % dt)

    eval_points = []
    targets = []

    # Phonemes must be sorted to match targets!
    phonemes = sorted(list(audio))
    for i, phoneme in enumerate(phonemes):
        simtime = 0.0  # Limit the amount of sim time per phoneme
        for sound in audio[phoneme]:
            sound_proc = ArrayProcess(sound.ravel())
            ph_target = np.zeros(len(phonemes))
            ph_target[i] = 1.0

            ep, targ = ph_traindata(model, detector, sound_proc, ph_target)
            eval_points.append(ep)
            targets.append(targ)
            simtime += sound.size * dt
            if simtime > detector.max_simtime:
                print "'%s' done" % phoneme
                break

    return np.concatenate(eval_points), np.concatenate(targets)


class TrainingData(object):
    def __init__(self, model, detector):
        self.model = model
        self.detector = detector

    @staticmethod
    def clear_cache():
        for fn in os.listdir(config.cache_dir):
            if fn.endswith(".npz"):
                os.remove(os.path.join(config.cache_dir, fn))

    @property
    def generated(self):
        return os.path.exists(self.cache_file())

    def generate(self):
        if self.generated:
            return

        audio = self.generate_audio()
        eval_points, targets = self.generate_traindata(audio)
        np.savez(file=self.cache_file(),
                 audio=audio,
                 eval_points=eval_points,
                 targets=targets)

    def generate_audio(self):
        return extract_all_audio(phonemes=self.detector.phonemes,
                                 rms=self.detector.rms,
                                 t_before=self.detector.t_before)

    def generate_traindata(self, audio):
        return generate_traindata(model=self.model,
                                  detector=self.detector,
                                  audio=audio)

    def cache_key(self):
        """Compute a key for the hash.

        This should be deterministic, and with a low probability of collisions.
        SHA1, as is done in Nengo, seems sufficient.
        """
        h = hashlib.sha1()
        h.update(dill.dumps(self.model))
        h.update(dill.dumps(self.detector))
        return h.hexdigest()

    def cache_file(self):
        return os.path.join(config.cache_dir, "%s.npz" % self.cache_key())

    def get(self):
        if not self.generated:
            raise RuntimeError("Training data not found; call generate first.")
        data = np.load(self.cache_file())
        return data['eval_points'], data['targets']
