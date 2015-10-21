import os
import shutil
import tarfile
from collections import defaultdict

from . import config

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

    @property
    def basename(self):
        return os.path.join(config.timit_root,
                            self.corpus,
                            "DR%d" % self.region,
                            "%s%s" % (self.sex, self.spkr_id),
                            "S%s%d" % (self.sent_type, self.sent_number))

    @property
    def wav(self):
        return "%s.WAV" % self.basename

    @property
    def phn(self):
        return "%s.PHN" % self.basename


def extract_audio(utterance, phonemes, frames_before=0):
    """Extract instances of the passed phonemes in the utterance."""
    ret = defaultdict(list)
    data, fs = sf.read(utterance.wav)
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


class TrainingData(object):
    def __init__(self, model, frames_before=0):
        self.model = model
        self.frames_before = frames_before

        # Try a few ways to limit it... Profile all!
        # 1. Just get all data
        # 2. Get all data and then only take N samples per phoneme
        # 3. Randomly sample utterances, keep going until N samples per phoneme
        # 4. Randomly sample utterances, keep going until M samples total

    def __hash__(self):
        # Figure out a hash function (look at nengo.cache?)
        pass

    def clear_cache(self):
        pass

    @property
    def eval_points(self):
        # If no cache, just return None or raise an error
        pass

    @property
    def targets(self):
        # If no cache, just return None or raise an error
        pass
