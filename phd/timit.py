import hashlib
import os
import shutil
import tarfile

import nengo
import nengo.utils.numpy as npext
import numpy as np
import soundfile as sf
from nengo.cache import Fingerprint

from . import config
from .networks.phonemes import connect_detector
from .processes import ArrayProcess

consonants = [
    'b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',
    'jh', 'ch',
    's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',
    'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',
    'l', 'r', 'w', 'y', 'hh', 'hv', 'el'
]

# "the closure intervals of stops which are distinguished from the stop release"
closures = {
    'bcl': 'b',
    'dcl': 'd',
    'gcl': 'g',
    'pcl': 'p',
    'tck': 't',
    'kcl': 'k',
    'dcl': 'jh',
    'tcl': 'ch',
}

vowels = [
    'iy', 'ih', 'eh', 'ey',
    'ae', 'aa', 'aw', 'ay', 'ah', 'ao',
    'oy', 'ow', 'uh', 'uw', 'ux',
    'er', 'ax', 'ix', 'axr', 'ax-h',
]

ignores = [
    'pau', 'epi', 'h#', '1', '2',
]

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


def extract_audio(utterance, phonemes, rms=0.5, frames_before=0):
    """Extract instances of the passed phonemes in the utterance."""
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


def extract_all_audio(phonemes, rms=0.5, frames_before=0):
    """Generate the audio sequences that will be used."""
    # Try a few ways to limit it...
    # 1. Just get all data
    # 2. Get all data and then only take N samples per phoneme
    # 3. Randomly sample utterances, keep going until N samples per phoneme
    # 4. Randomly sample utterances, keep going until M samples total

    # Let's get all data for now (in the training corpus)
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
                                               frames_before))
    return audio


def generate_eval_points(model, derivatives, audio,
                         max_simtime=5.0, sample_every=0.001):
    """Simulate Sermo with the extracted audio."""
    assert np.allclose(sample_every / dt, int(sample_every / dt)), (
        "sample_every must be a multiple of dt (%s)" % dt)

    eval_points = []
    dims = model.recognition.periphery.freqs.size
    size_in = dims * (len(derivatives) + 1)

    # Phonemes must be sorted to match targets!
    phonemes = sorted(list(audio))

    # TODO: Parallelize this
    for i, phoneme in enumerate(phonemes):
        sound = ArrayProcess(np.concatenate(audio[phoneme]).ravel())
        net = model.build()
        # Put in our own sound and fs in the built network
        net.periphery.sound_process = sound
        net.periphery.fb.sound_process = sound
        net.periphery.fs = fs
        net.periphery.fb.samplerate = fs
        with net:
            # Aggregate all the data needed
            ph_inputs = nengo.Node(size_in=size_in)
            connect_detector(net, derivatives, ph_inputs)
            pr = nengo.Probe(ph_inputs,
                             synapse=0.01, sample_every=sample_every)
        sim = nengo.Simulator(net, dt=dt)
        sim.run(min(dt * sound.array.size, max_simtime))
        eval_points.append(sim.data[pr])

    return np.concatenate(eval_points)


def generate_targets(audio,
                     max_simtime=5.0, sample_every=0.001, frames_before=0):
    targets = []
    sample_rate = 1. / sample_every
    assert np.allclose(fs / sample_rate, int(fs / sample_rate)), (
        "Sample rate must be a multiple of fs (%s)" % fs)
    step = int(fs / sample_rate)
    steps_before = frames_before // step

    # Phonemes must be sorted to match eval_points!
    phonemes = sorted(list(audio))
    n_phonemes = len(phonemes)
    max_samples = int(max_simtime // sample_every) + 1

    for i, phoneme in enumerate(phonemes):
        ph_targets = []
        for sound in audio[phoneme]:
            target = np.zeros((n_phonemes, int(sound.size // step)))
            target[i, steps_before:] = 1
            ph_targets.append(target)

        # Each phoneme is limited to max_simtime
        ph_targets = np.concatenate(ph_targets, axis=1)
        if ph_targets.shape[1] > max_samples:
            ph_targets = ph_targets[:, :max_samples]
        targets.append(ph_targets)

    return np.concatenate(targets, axis=1)

class TrainingData(object):
    def __init__(self, model, derivatives, phonemes,
                 rms=0.5, max_simtime=5.0, sample_every=0.001):
        self.model = model
        self.derivatives = derivatives
        self.phonemes = phonemes
        self.rms = rms
        self.max_simtime = max_simtime
        self.sample_every = sample_every
        self.frames_before = int(fs * max(derivatives))

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
        eval_points = self.generate_eval_points(audio)
        targets = self.generate_targets(audio)

        np.savez(file=self.cache_file(),
                 audio=audio,
                 eval_points=eval_points,
                 targets=targets)

    def generate_audio(self):
        return extract_all_audio(phonemes=self.phonemes,
                                 rms=self.rms,
                                 frames_before=self.frames_before)

    def generate_eval_points(self, audio):
        return generate_eval_points(model=self.model,
                                    derivatives=self.derivatives,
                                    audio=audio,
                                    max_simtime=self.max_simtime,
                                    sample_every=self.sample_every)

    def generate_targets(self, audio):
        return generate_targets(
            audio, self.max_simtime, self.sample_every, self.frames_before)

    def cache_key(self):
        """Compute a key for the hash.

        This should be deterministic, and with a low probability of collisions.
        SHA1, as is done in Nengo, seems sufficient.
        """
        h = hashlib.sha1()
        h.update(str(Fingerprint(self.model)))
        h.update(str(Fingerprint(self.derivatives)))
        h.update(str(Fingerprint(self.phonemes)))
        h.update(str(Fingerprint(self.sample_every)))
        h.update(str(Fingerprint(self.rms)))
        h.update(str(Fingerprint(self.frames_before)))
        return h.hexdigest()

    def cache_file(self):
        return os.path.join(config.cache_dir, "%s.npz" % self.cache_key())

    def get(self):
        if not self.generated:
            raise RuntimeError("Training data not found; call generate first.")
        data = np.load(self.cache_file())
        return data['eval_points'], data['targets']
