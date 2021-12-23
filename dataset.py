"""Preprocessing for the MusicNet dataset."""
import h5py
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
from scipy.signal import spectrogram

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

# Normalize to major key, accounting for variation in naming schemes
# within the MusicNet metadata.
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
                        samples_per_chunk: int) -> List[np.ndarray]:
    """Breaks fingerprints of a recording into fixed-length chunks."""
    chunks = []
    for pos in range(0, len(fingerprints), samples_per_chunk):
        chunk = fingerprints[pos:pos + samples_per_chunk]
        # exclude partial chunks (at end)
        if chunk.shape[0] == samples_per_chunk:
            chunks.append(chunk)
    return chunks


@dataclass
class MusicNet:
    """Preprocessing for the MusicNet dataset."""
    dataset_meta_path: str
    dataset_path: Optional[str]
    fingerprints_cache_path: Optional[str] = None
    fs: int = 44100
    window_size: int = 2048
    window_overlap: int = 512
    n_features: int = 128
    chunk_size_sec: float = 10

    def __post_init__(self):
        self.samples_per_chunk = int(self.fs * self.chunk_size_sec /
                                     (self.window_size - self.window_overlap))
        if self.dataset_path:
            self.dataset = h5py.File(self.dataset_path, 'r')
        else:
            logging.info(
                'No raw dataset path specified. Using cached fingerprints.')
            self.dataset = None
        self.meta_df = pd.read_csv(self.dataset_meta_path).set_index('id')
        self._preprocess_key()
        self._preprocess_ensemble()
        self._load_fingerprints_cache()

    def recordings_by_column(
            self,
            col: str) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, int]]:
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
        return recordings, np.array(recording_label_ids), label_to_id

    def chunks_by_column(
            self, col: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Returns audio fingerprint chunks grouped by metadata column."""
        chunks = []
        chunk_label_ids = []
        recording_fingerprints, label_ids, label_to_id = self.recordings_by_column(
            col)
        for (fingerprints, label_id) in zip(recording_fingerprints, label_ids):
            recording_chunks = recording_to_chunks(fingerprints,
                                                   self.samples_per_chunk)
            chunks += recording_chunks
            chunk_label_ids += [label_id] * len(recording_chunks)
        return np.array(chunks), np.array(chunk_label_ids), label_to_id

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
        fingerprint_indices = np.geomspace(
            1, self.window_size // 2 + 1,
            self.n_features).round().astype(int) - 1
        for key in tqdm(self.dataset):
            _, _, audio_fingerprint = spectrogram(self.dataset[key]['data'][:],
                                                  nperseg=self.window_size,
                                                  noverlap=self.window_overlap)
            fingerprints_by_id[key.split('id_')
                               [1]] = audio_fingerprint[fingerprint_indices].T
        return fingerprints_by_id

    def _load_fingerprints_cache(self):
        """Loads or generates (and saves) audio fingerprints."""
        if self.fingerprints_cache_path:
            try:
                self.fingerprints_by_id = np.load(self.fingerprints_cache_path)
                logging.info('Loaded fingerprints from cache.')
            except FileNotFoundError:
                self.fingerprints_by_id = self._generate_fingerprints()
                np.savez_compressed(self.fingerprints_cache_path,
                                    **self.fingerprints_by_id)
        else:
            logging.warning('No fingerprint cache path specified.')
            self.fingerprints_by_id = self._generate_fingerprints()
