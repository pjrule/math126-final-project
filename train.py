"""CLI wrapper for audio classification models."""
import logging
import click
import joblib
import numpy as np
import pandas as pd
from dataset import recording_to_chunks
import xgboost as xgb
from typing import Optional, Tuple
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from dataset import MusicNet, chunks_to_samples


def init_xgboost(random_state: int, verbosity: int) -> xgb.XGBClassifier:
    """Initializes an XGBoost model for audio classification."""
    return xgb.XGBClassifier(random_state=random_state,
                             use_label_encoder=False,
                             objective='multi:softprob',
                             eval_metric='auc',
                             subsample=0.5,
                             verbosity=verbosity)


def init_svc(random_state: int, verbosity: int) -> LinearSVC:
    """Initializes a linear SVC model for audio classification."""
    return LinearSVC(random_state=random_state,
                     verbose=verbosity,
                     max_iter=5000)


def init_ksvd(random_state: int, verbosity: int) -> None:
    """Initializes a KSVD model for audio classification."""
    raise NotImplementedError('not implemented yet! :(')


def init_lu(random_state: int, verbosity: int) -> None:
    """Initializes a randomized LU model for audio classification."""
    raise NotImplementedError('not implemented yet! :(')


MODELS = {
    'xgboost': init_xgboost,
    'svc': init_svc,
    'ksvd': init_ksvd,
    'lu': init_lu
}

try:
    grad_model = joblib.load('grad_model_ensemble.joblib')
except FileNotFoundError:
    joblib.dump(grad_model, 'grad_model_ensemble.joblib')


def training_set(dataset: MusicNet, column: str, split_by: str,
                 test_split: float,
                 random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a training set of fingerprints rom the MusicNet dataset.

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
        A tuple containing a matrix of fingerprints and associated labels.
    """
    shuffle_rng = np.random.default_rng(random_state)
    if split_by == 'chunk':
        chunks, chunk_labels = dataset.chunks_by_column(column)
        chunks_train, _, chunk_labels_train, _ = train_test_split(
            chunks,
            chunk_labels,
            test_size=test_split,
            random_state=random_state)
        samples_train, sample_labels_train = chunks_to_samples(
            chunks_train, chunk_labels_train, shuffle=True, rng=shuffle_rng)
    elif split_by == 'recording':
        recordings, recording_labels = dataset.recordings_by_column(column)
        recordings_train, _, recording_labels_train, _ = train_test_split(
            recordings,
            recording_labels,
            test_size=test_split,
            random_state=random_state)
        chunks_train = recording_to_chunks(recordings_train,
                                           dataset.intervals_per_chunk)
        chunk_labels_train = np.repeat(recording_labels, chunks_train.shape[1])
        samples_train, sample_labels_train = chunks_to_samples(
            chunks_train, chunk_labels_train, shuffle=True, rng=shuffle_rng)
    elif split_by == 'sample':
        chunks, chunk_labels = dataset.chunks_by_column(column)
        samples, sample_labels = chunks_to_samples(chunks,
                                                   chunk_labels,
                                                   shuffle=True,
                                                   rng=shuffle_rng)
        samples_train, _, sample_labels_train, _ = train_test_split(
            chunks,
            chunk_labels,
            test_size=test_split,
            random_state=random_state)
    else:
        raise ValueError(f'Unsupported splitting mode "{split_by}".')
    return samples_train, sample_labels_train


@click.command()
@click.option('--dataset-path',
              help='Path of the MusicNet dataset (HDF5 format).')
@click.option('--dataset-meta-path',
              help='Path of the MusicNet metadata (CSV format).',
              required=True)
@click.option('--fingerprints-cache-path',
              help='Path of the audio fingerprint cache.')
@click.option('--out-path', help='Output path for the trained model.')
@click.option('--column',
              required=True,
              help='Metadata column to classify on.')
@click.option('--model', required=True, help='Type of model to train.')
@click.option('--split-by', default='chunk')
@click.option('--random-state',
              type=int,
              default=0,
              help='Random seed for models and test/train splitting.')
@click.option('--test-split',
              type=click.FloatRange(0, 1),
              help='Proportion of samples to save for testing.')
@click.option('--train-subsample-size',
              type=int,
              help='Absolute number of samples to use for training.')
@click.option('--verbose',
              type=int,
              default=100,
              help='Verbosity level for training (forwarded to scikit-learn).')
def main(dataset_path: Optional[str], dataset_meta_path: str,
         fingerprints_cache_path: Optional[str], out_path: Optional[str],
         column: str, model: str, split_by: str, random_state: int,
         test_split: float, train_subsample_size: int, verbose: int):
    """Trains an audio classification model."""
    if column not in ('composer', 'key', 'ensemble'):
        raise ValueError(f'Unsupported classification column "{column}.')

    logging.info('Loading MusicNet dataset...')
    dataset = MusicNet(dataset_path=dataset_path,
                       dataset_meta_path=dataset_meta_path,
                       fingerprints_cache_path=fingerprints_cache_path)
    samples_train, sample_labels_train = training_set(
        dataset=dataset,
        column=column,
        split_by=split_by,
        test_split=test_split,
        random_state=random_state)

    try:
        init_fn = MODELS[model]
    except KeyError:
        raise ValueError(f'Unsupported model type "{model}.')
    model = init_fn(random_state, verbose)

    logging.info('Fitting model...')
    Path(out_path).touch()
    if train_subsample_size is None:
        model.fit(samples_train, sample_labels_train)
    else:
        # We assume samples are shuffled.
        model.fit(samples_train[:train_subsample_size],
                  sample_labels_train[:train_subsample_size])
    joblib.dump(model, out_path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    main()
