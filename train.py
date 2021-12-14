"""CLI wrapper for audio classification models."""
import logging
import click
import joblib
import numpy as np
import xgboost as xgb
from typing import Optional
from pathlib import Path
from sklearn.svm import LinearSVC
from dataset import MusicNet
from split import train_test_split
from models import SVDClassifier, RandomizedLUClassifier
from time import time


def init_xgboost(random_state: int, verbosity: int) -> xgb.XGBClassifier:
    """Initializes an XGBoost model for audio classification."""
    return xgb.XGBClassifier(random_state=random_state,
                             use_label_encoder=False,
                             objective='multi:softprob',
                             eval_metric='auc',
                             subsample=0.5,
                             max_depth=3,   
                             verbosity=verbosity)


def init_svc(random_state: int, verbosity: int) -> LinearSVC:
    """Initializes a linear SVC model for audio classification."""
    return LinearSVC(random_state=random_state,
                     verbose=verbosity,
                     max_iter=5000)


def init_svd(random_state: int, verbosity: int) -> None:
    """Initializes a PCA model for audio classification."""
    return SVDClassifier(k=32, random_state=random_state, verbosity=verbosity)


def init_lu(random_state: int, verbosity: int) -> None:
    """Initializes a randomized LU model for audio classification."""
    return RandomizedLUClassifier(k=32,
                                  random_state=random_state,
                                  verbosity=verbosity)


MODELS = {
    'xgboost': init_xgboost,
    'svc': init_svc,
    'svd': init_svd,
    'lu': init_lu
}


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
@click.option('--split-by', default='recording')
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
              default=3,
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
    samples_train, _, sample_labels_train, _ = train_test_split(
        dataset=dataset,
        column=column,
        split_by=split_by,
        test_split=test_split,
        random_state=random_state)

    try:
        init_fn = MODELS[model]
    except KeyError:
        raise ValueError(f'Unsupported model type "{model}.')
    classifier = init_fn(random_state, verbose)

    logging.info('Fitting classifier...')
    Path(out_path).touch()
    logging.info('Training classifier "%s" on column "%s".', classifier, column)
    logging.info('Classifier details:', model)
    tic = time()
    if train_subsample_size is None:
        classifier.fit(samples_train, sample_labels_train)
    else:
        # We assume samples are shuffled.
        classifier.fit(samples_train[:train_subsample_size],
                  sample_labels_train[:train_subsample_size])
    logging.info('Trained classifier "%s" on column "%s" in %.2f seconds.',
        classifier,
        column,
        time() - tic) 
    joblib.dump(model, out_path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    main()
