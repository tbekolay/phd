import os
import shutil
import tarfile
import warnings

import numpy as np
import soundfile as sf
from nengo.utils import numpy as npext

from .utils import ensuretuple


class Utterance(object):
    def __init__(self,
                 root, corpus, region, sex, spkr_id, sent_type, sent_number):
        self.root = root
        self.corpus = corpus
        self.region = region
        self.sex = sex
        self.spkr_id = spkr_id
        self.sent_type = sent_type
        self.sent_number = sent_number
        self._n_frames = None

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
        root, corpus = os.path.split(corpus_dir)
        return cls(root, corpus, region, sex, spkr_id, sent_type, sent_number)

    @property
    def path(self):
        return os.path.join(self.root,
                            self.corpus,
                            "DR%d" % self.region,
                            "%s%s" % (self.sex, self.spkr_id),
                            "S%s%d" % (self.sent_type, self.sent_number))

    @property
    def n_frames(self):
        if self._n_frames is not None:
            return self._n_frames

        with open("%s.PHN" % self.path, 'r') as phnfile:
            line = phnfile.readlines()[-1]
        self._n_frames = int(line.split()[1])
        return self._n_frames

    @staticmethod
    def _parse_text(path):
        substitutions = TIMIT.substitutions
        info = []
        with open(path, 'r') as fp:
            for line in fp:
                start, end, lbl = line.split()
                lbl = substitutions[lbl] if lbl in substitutions else lbl
                info.append({'start': int(start), 'end': int(end), 'lbl': lbl})
        return info

    def _get_samples(self, path, labels, rms=0.5):
        data, fs = sf.read("%s.WAV" % self.path)
        assert fs == TIMIT.fs, "fs (%s) != 16000" % fs
        # Normalize data to desired rms (or do this per sample?)
        data_rms = npext.rms(data)
        data *= (rms / data_rms)
        info = self._parse_text(path)
        samples = {lbl: [] for lbl in labels}
        for inf in info:
            if inf['lbl'] in samples:
                samples[inf['lbl']].append(data[inf['start']: inf['end']])
        return samples

    def _get_targets(self, path, labels):
        info = self._parse_text(path)
        target = np.zeros((self.n_frames, len(labels)))
        for inf in info:
            if inf['lbl'] in labels:
                idx = labels.index(inf['lbl'])
                target[inf['start']: inf['end'], idx] = 1.0
        return target

    def _get_idx(self, path, labels):
        info = self._parse_text(path)
        target = np.zeros((self.n_frames,), dtype=bool)
        for inf in info:
            if inf['lbl'] in labels:
                target[inf['start']: inf['end']] = True
        return target

    def audio(self, words=None, phonemes=None, rms=0.5):
        """Returns the audio waveform, normalized to `rms`."""
        data, fs = sf.read("%s.WAV" % self.path)
        assert fs == TIMIT.fs, "fs (%s) != 16000" % fs
        # Normalize data to desired rms
        data_rms = npext.rms(data)
        data *= (rms / data_rms)
        # Sometimes the audio is longer than the PHN file.
        # So, we will clip the end of the 'audio' if that's the case.
        # This may waste some time opening and reading this file,
        # but reading this text file is much faster than reading the
        # audio file and editing the 'phn' target accordingly.
        data = data[:self.n_frames]
        if words is not None:
            return data[self.word_idx(words)]
        elif phonemes is not None:
            return data[self.phn_idx(phonemes)]
        return data

    def phn_idx(self, phonemes):
        return self._get_idx("%s.PHN" % self.path, phonemes)

    def word_idx(self, words):
        return self._get_idx("%s.WRD" % self.path, words)

    def phn(self, phonemes):
        """Returns the phoneme target array."""
        phn = self._get_targets("%s.PHN" % self.path, phonemes)
        return phn[self.phn_idx(phonemes)]

    def word(self, words):
        """Returns the word target array."""
        wrd = self._get_targets("%s.WRD" % self.path, words)
        return wrd[self.word_idx(words)]

    def phn_samples(self, phonemes, rms=0.5):
        return self._get_samples("%s.PHN" % self.path, phonemes, rms)

    def word_samples(self, words, rms=0.5):
        return self._get_samples("%s.WRD" % self.path, words, rms)


class FileFilter(object):
    def __init__(self, region=None, sex=None, spkr_id=None, sent_type=None,
                 sent_number=None):
        self.region = region
        self.sex = sex
        self.spkr_id = spkr_id
        self.sent_type = sent_type
        self.sent_number = sent_number

    def matches(self, utterance):
        attrs = ('region', 'sex', 'spkr_id', 'sent_type', 'sent_number')

        def _matchattr(attr):
            val = getattr(self, attr)
            if val is not None:
                if getattr(utterance, attr) not in ensuretuple(val):
                    return False
            return True

        return all(_matchattr(attr) for attr in attrs)

    def iter_utterances(self, datadir, corpus="train"):
        corpus_d = os.path.join(datadir, corpus.upper())
        for region_f in os.listdir(corpus_d):
            region_d = os.path.join(corpus_d, region_f)
            for spkr_f in os.listdir(region_d):
                spkr_d = os.path.join(region_d, spkr_f)
                for utt_f in os.listdir(spkr_d):
                    if utt_f.endswith('.WAV'):
                        utt_path = os.path.join(spkr_d, utt_f[:-4])
                        utt = Utterance.from_path(utt_path)
                        if self.matches(utt):
                            yield utt


class TIMIT(object):
    """An interface to TIMIT for machine learning experiments.

    Parameters
    ----------
    datadir : str
        Path pointing to the directory where TIMIT data is stored,
        or will be stored when the data is extracted.
    """

    consonants = ['b', 'd', 'g', 'p', 't', 'k', 'dx',
                  'jh', 'ch',
                  's', 'sh', 'z', 'f', 'th', 'v', 'dh',
                  'm', 'n', 'ng',
                  'l', 'r', 'w', 'y', 'hh']

    vowels = ['iy', 'ih', 'eh', 'ey',
              'ae', 'aa', 'aw', 'ay', 'ah',
              'oy', 'ow', 'uh', 'uw',
              'er']

    silence = ['sil']

    phones = consonants + vowels + silence

    # Substitute so that we have 38 phones + silence in the end;
    # see "Phone Recognition on the TIMIT Database" for details.
    substitutions = {'ao': 'aa',
                     'ax': 'ah',
                     'ax-h': 'ah',
                     'axr': 'er',
                     'hv': 'hh',
                     'ix': 'ih',
                     'el': 'l',
                     'em': 'm',
                     'en': 'n',
                     'nx': 'n',
                     'eng': 'ng',
                     'zh': 'sh',
                     'ux': 'uw',
                     'pcl': 'sil',
                     'tcl': 'sil',
                     'kcl': 'sil',
                     'bcl': 'sil',
                     'dcl': 'sil',
                     'gcl': 'sil',
                     'h#': 'sil',
                     'pau': 'sil',
                     'epi': 'sil'}

    fs = 16000  # TIMIT is always 16 kHz
    dt = 1. / fs

    def __init__(self, datadir):
        if datadir.endswith('timit'):
            datadir, _ = os.path.split(datadir)
        self.datadir = os.path.abspath(os.path.expanduser(datadir))
        self.filefilt = FileFilter()

    @property
    def timitdir(self):
        return os.path.join(self.datadir, 'timit')

    def untar(self, tgz_path, overwrite=False):
        """Extract the file at `tgz_path` to `datadir`."""
        if not overwrite and os.path.exists(self.timitdir):
            raise IOError("'timit' directory already exists. Pass "
                          "'overwrite=True' if you want to untar anyway.")

        tgz_path = os.path.abspath(os.path.expanduser(tgz_path))

        # Let's just extract it all and clean up the paths afteward
        with tarfile.open(tgz_path) as timit_tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(timit_tar, path=self.datadir)

        # We don't need the README or the CONVERT or SPHERE directories
        shutil.rmtree(os.path.join(self.timitdir, 'CONVERT'))
        shutil.rmtree(os.path.join(self.timitdir, 'SPHERE'))
        os.remove(os.path.join(self.timitdir, 'README.DOC'))

        # Then, let's move what's left to the root 'timit' directory
        def mvtotimitroot(fileordir):
            shutil.move(os.path.join(self.timitdir, 'TIMIT', fileordir),
                        os.path.join(self.timitdir, fileordir))
        mvtotimitroot('TEST')
        mvtotimitroot('TRAIN')
        mvtotimitroot('README.DOC')

        # Move all docs to the root too
        docdir = os.path.join(self.timitdir, 'TIMIT', 'DOC')
        for doc in os.listdir(docdir):
            shutil.move(os.path.join(docdir, doc),
                        os.path.join(self.timitdir, doc))

        # Remove now-empty TIMIT dir
        shutil.rmtree(os.path.join(self.timitdir, 'TIMIT'))

    def word_samples(self, words, rms=0.5, corpus="train"):
        samples = {word: [] for word in words}
        for utt in self.filefilt.iter_utterances(self.timitdir, corpus):
            sample = utt.word_samples(words, rms=rms)
            for word in words:
                samples[word].extend(sample[word])
        for word in words:
            if len(samples[word]) == 0:
                del samples[word]
                warnings.warn("Word '%s' not found in corpus." % word)
        return samples

    def phn_samples(self, phonemes, rms=0.5, corpus="train"):
        samples = {phone: [] for phone in phonemes}
        for utt in self.filefilt.iter_utterances(self.timitdir, corpus):
            sample = utt.phn_samples(phonemes, rms=rms)
            for phone in phonemes:
                samples[phone].extend(sample[phone])
        for phone in phonemes:
            if len(samples[phone]) == 0:
                del samples[phone]
                warnings.warn("Phone '%s' not found in corpus." % phone)
        return samples

    def in_audio(self, words=None, phonemes=None):
        audio = []
        for utt in self.filefilt.iter_utterances(self.timitdir):
            audio.append(utt.audio(words=words, phonemes=phonemes))
        return np.concatenate(audio)

    def out_phn(self, phonemes):
        phn = []
        for utt in self.filefilt.iter_utterances(self.timitdir):
            phn.append(utt.phn(phonemes))
        return np.concatenate(phn)

    def out_word(self, words):
        word = []
        for utt in self.filefilt.iter_utterances(self.timitdir):
            word.append(utt.word(words))
        return np.concatenate(word)
