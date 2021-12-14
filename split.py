"""Utility functions for splitting datasets."""
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split as sk_split
from dataset import recording_to_chunks, chunks_to_samples, MusicNet


def train_test_split(
    dataset: MusicNet, column: str, split_by: str, test_split: float,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits the MusicNet dataset into training and test sets.

    We support three modes for splitting the dataset.
      * `chunk` (default) -- recording fingerprints are broken into
        fixed-length chunks. These chunks are then shuffled and split
        into training and test sets. For training, we break chunks up
        further into individual samples.
      * `recording` -- recording fingerprints are broken into
        training and test sets, which are then separately broken into
        chunks (and then samples).
      * `samples` -- recording fingerprint are broken into individual
        samples before forming training and test sets.

    We generally train models at the sample level and evaluate models
    by aggregating predictions over chunks. Splitting by recording theoretically
    reduces leak between the training and test sets, but because there are
    only 330 recordings in the MusicNet dataset, it may be reasonable
    to split at the chunk level.

    Args:
        dataset: MusicNet dataset wrapper. 
        column: Metadata column to generate 0-indexed labels from.
        split_by: Splitting mode (`chunk`, `recording`, or `sample`).
        test_split: Approximate proportion of data (in units specified by
          `split_by`) to save for model evaluation.
        random_state: Random seed for splitting and shuffling. 

    Returns:
        A tuple containing a matrix of fingerprints and associated labels
        in the order (training data, training labels, test data, test labels).
    """
    shuffle_rng = np.random.default_rng(random_state)
    if split_by == 'chunk':
        chunks, chunk_labels = dataset.chunks_by_column(column)
        chunks_train, chunks_test, chunk_labels_train, chunk_labels_test = sk_split(
            chunks,
            chunk_labels,
            test_size=test_split,
            random_state=random_state)
        samples_train, sample_labels_train = chunks_to_samples(
            chunks_train, chunk_labels_train, shuffle=True, rng=shuffle_rng)
        samples_test, sample_labels_test = chunks_to_samples(chunks_test,
                                                             chunk_labels_test,
                                                             shuffle=True,
                                                             rng=shuffle_rng)
    elif split_by == 'recording':
        recordings, recording_labels = dataset.recordings_by_column(column)
        recordings_train, recordings_test, recording_labels_train, recording_labels_test = sk_split(
            recordings,
            recording_labels,
            test_size=test_split,
            random_state=random_state)
        chunks_train = []
        chunk_labels_train = []
        for recording, label in zip(recordings_train, recording_labels_train):
            recording_chunks = recording_to_chunks(recording,
                                                   dataset.intervals_per_chunk)
            chunks_train += recording_chunks
            chunk_labels_train += [label] * len(recording_chunks)
        chunks_test = []
        chunk_labels_test = []
        for recording, label in zip(recordings_test, recording_labels_test):
            recording_chunks = recording_to_chunks(recording,
                                                   dataset.intervals_per_chunk)
            chunks_test += recording_chunks
            chunk_labels_test += [label] * len(recording_chunks)

        samples_train, sample_labels_train = chunks_to_samples(
            np.array(chunks_train),
            np.array(chunk_labels_train),
            shuffle=True,
            rng=shuffle_rng)
        samples_test, sample_labels_test = chunks_to_samples(
            np.array(chunks_test),
            np.array(chunk_labels_test),
            shuffle=True,
            rng=shuffle_rng)
    elif split_by == 'sample':
        chunks, chunk_labels = dataset.chunks_by_column(column)
        samples, sample_labels = chunks_to_samples(chunks,
                                                   chunk_labels,
                                                   shuffle=True,
                                                   rng=shuffle_rng)
        samples_train, samples_test, sample_labels_train, sample_labels_test, = sk_split(
            chunks,
            chunk_labels,
            test_size=test_split,
            random_state=random_state)
    else:
        raise ValueError(f'Unsupported splitting mode "{split_by}".')
    return samples_train, samples_test, sample_labels_train, sample_labels_test
