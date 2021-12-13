"""Preprocessing for the MusicNet dataset."""
import h5py
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
from scipy.fft import fft
from scipy.signal.windows import hamming

ENSEMBLE_CATEGORIES = {
    'Piano Ensemble': ('Piano Trio', 'Piano Quartet', 'Piano Quintet'),
    'Wind Ensemble': ('Pairs Clarinet-Horn-Bassoon', 'Clarinet Quintet',
                      'Wind Octet', 'Wind Quintet'),
    'String Ensemble': ('String Quartet', 'String Sextet', 'Viola Quintet'),
    # <9 occurrences each
    'Other': ('Violin and Harpsichord', 'Wind and Strings Octet',
              'Accompanied Cello', 'Accompanied Clarinet', 'Solo Flute',
              'Clarinet-Cello-Piano Trio', 'Horn Piano Trio')
}

# normalize to major key
CIRCLE_OF_FIFTHS = {
    'A major': 'A',
    'B-flat major': 'Bb',
    'C major': 'C',
    'A minor': 'C',
    'D-flat major': 'Db',
    'D major': 'D',
    'G major': 'G',
    'E-flat major': 'Eb',
    'D minor': 'F',
    'E major': 'E',
    'F major': 'F',
    'B-flat major': 'Bb',
    'D Minor': 'F',
    'F Major': 'F',
    'E minor': 'G',
    'F minor': 'Ab',
    'D Major': 'D',
    'G minor': 'Bb',
    'F': 'F',
    'F-sharp major': 'Gb',
    'C-sharp major': 'Db',
    'B-flat minor': 'Db',
    'C-sharp minor': 'E',
    'B major': 'B',
    'A-flat major': 'Ab',
    'B minor': 'D',
    'E-flat minor': 'Gb',
    'F-sharp minor': 'A',
    'G-sharp minor': 'B',
    'C minor': 'Eb',
    'B-flat Major': 'Bb'
}


def audio_fingerprint(audio: np.ndarray, sample_interval: int,
                      window_size: int):
    """Takes a fingerprint of an audio signal by sampling the spectogram at a
  fixed interval, normalizing each sample, and filtering out high frequencies.
 
  We assume that the audio is sampled at roughly 40 kHz (44.1 kHz and
  48 kHz are common), such that returning the first quarter of the spectrogram
  truncates around 5 kHz.
  """
    n_samples = audio.shape[0] // sample_interval
    fingerprint = np.empty((n_samples, window_size // 8))
    window = hamming(window_size)
    for sample_idx in range(n_samples):
        sample = audio[sample_idx * window_size:(sample_idx + 1) * window_size]
        sample_mag = np.abs(sample)
        if sample_mag.max() > 0:
            normalized_sample = sample / sample_mag.max()
        else:
            normalized_sample = sample
        windowed_sample = window * normalized_sample
        fingerprint[sample_idx] = np.abs(fft(windowed_sample))[:window_size //
                                                               8]
    return fingerprint


def chunks_to_samples(chunks: np.ndarray,
                      chunk_labels: np.ndarray,
                      shuffle: bool = False,
                      rng: Optional[np.random.RandomState] = None):
    """Flattens audio chunks to individual audio samples."""
    samples = chunks.reshape(chunks.shape[0] * chunks.shape[1], -1)
    sample_labels = chunk_labels.repeat(chunks.shape[1])
    if shuffle:
        indices = np.arange(sample_labels.size)
        if rng:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)
        return samples[indices], sample_labels[indices]
    return samples, sample_labels


def recording_to_chunks(fingerprints: np.ndarray,
                        intervals_per_chunk: int) -> List[np.ndarray]:
    """Breaks fingerprints of a recording into fixed-length chunks."""
    chunks = []
    for pos in range(0, len(fingerprints), intervals_per_chunk):
        chunk = fingerprints[pos:pos + intervals_per_chunk]
        # exclude partial chunks (at end)
        if chunk.shape[0] == intervals_per_chunk:
            chunks.append(chunk)
    return chunks


@dataclass
class MusicNet:
    """Preprocessing for the MusicNet dataset."""
    dataset_meta_path: str
    dataset_path: Optional[str]
    fingerprints_cache_path: Optional[str] = None
    fs: int = 44100
    window_size: int = 1024
    sample_interval_sec: float = 0.05
    chunk_size_sec: float = 10

    def __post_init__(self):
        self.sample_interval = int(self.sample_interval_sec * self.fs)
        self.intervals_per_chunk = int(self.chunk_size_sec /
                                       self.sample_interval_sec)
        self.dataset = h5py.File(self.dataset_path, 'r')
        self.meta_df = pd.read_csv(self.dataset_meta_path).set_index('id')
        self._preprocess_key()
        self._preprocess_ensemble()
        self._load_fingerprints_cache()

    def recordings_by_column(self,
                             col: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """Returns audio fingerprint recordings grouped by metadata column."""
        ids_by_col = {
            label: set(self.meta_df.iloc[idx].name for idx in indices)
            for label, indices in self.meta_df.groupby(col).indices.items()
        }
        label_to_id = {label: idx for idx, label in enumerate(ids_by_col)}

        recordings = []
        recording_label_ids = []
        for label, ids in ids_by_col.items():
            for recording_id in ids:
                recordings.append(self.fingerprints_by_id[str(recording_id)])
                recording_label_ids.append(label_to_id[label])
        return recordings, np.array(recording_label_ids)

    def chunks_by_column(self, col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns audio fingerpritn chunks grouped by metadata column."""
        chunks = []
        chunk_label_ids = []
        for (fingerprints, label_id) in zip(*self.recordings_by_column(col)):
            recording_chunks = recording_to_chunks(fingerprints,
                                                   self.intervals_per_chunk)
            chunks.append(recording_chunks)
            chunk_label_ids += [label_id] * recording_chunks.shape[0]
        return np.array(chunks), np.array(chunk_label_ids)

    def _preprocess_key(self):
        """Extracts (normalized) keys from metadata."""
        self.meta_df['key'] = (self.meta_df['composition'].str.split(
            ' in ').str[-1].str.split(' for ').str[0])
        self.meta_df['key'] = self.meta_df['key'].apply(
            lambda k: CIRCLE_OF_FIFTHS.get(k, k))

        # special case: Bach cello suites
        # TODO (@Eric): is this accurate?
        # pulled from Wikipedia, which claims that "the second bourrée,
        # though in C minor, has a two-flat (or G minor) key signature."
        self.meta_df.loc[self.meta_df['composition'] == 'Cello Suite 3',
                         'key'] = 'C'
        self.meta_df.loc[self.meta_df['composition'] == 'Cello Suite 4',
                         'key'] = 'Eb'

        # special case: 4 Impromptus
        self.meta_df.loc[self.meta_df['composition'] == '4 Impromptus',
                         'key'] = (self.meta_df.loc[
                             self.meta_df['composition'] == '4 Impromptus',
                             'movement'].str.split(' in ').str[1].apply(
                                 lambda k: CIRCLE_OF_FIFTHS[k]))

    def _preprocess_ensemble(self):
        """Simplifies ensemble metadata."""
        ensemble_mapping = {}
        for k, cats in ENSEMBLE_CATEGORIES.items():
            for cat in cats:
                ensemble_mapping[cat] = k
        self.meta_df['ensemble'] = self.meta_df['ensemble'].apply(
            lambda k: ensemble_mapping.get(k, k))

    def _generate_fingerprints(self) -> Dict[str, np.ndarray]:
        """Generates audio fingerprints."""
        logging.info('Generating audio fingerprints...')
        fingerprints_by_id = {}
        for key in tqdm(self.dataset):
            fingerprints_by_id[key.split('id_')[1]] = audio_fingerprint(
                audio=self.dataset[key]['data'][:],
                sample_interval=self.sample_interval,
                window_size=self.window_size)
        return fingerprints_by_id

    def _load_fingerprints_cache(self):
        """Loads or generates (and saves) audio fingerprints."""
        if self.fingerprints_cache_path:
            try:
                self.fingerprints_by_id = np.load(self.fingerprints_cache_path)
            except FileNotFoundError:
                self.fingerprints_by_id = self._generate_fingerprints()
                np.savez_compressed(self.fingerprints_cache_path,
                                    **self.fingerprints_by_id)
        else:
            logging.warning('No fingerprint cache path specified.')
            self.fingerprints_by_id = self._generate_fingerprints()
