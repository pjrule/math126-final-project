"""TODO"""
import logging
import click
import joblib
import numpy as np
from typing import Optional
from pathlib import Path
from sklearn.svm import LinearSVC
from dataset import MusicNet
from split import train_test_split
from models import SVDClassifier, RandomizedLUClassifier
from time import time


def chunk_predict(chunk, model):
    """Predicts the label for a chunk by voting (soft if possible)."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(chunk).sum(axis=0).argmax()
    return model.predict(chunk).sum(axis=0).argmax()


@click.command()
@click.option('--dataset-path',
              help='Path of the MusicNet dataset (HDF5 format).')
@click.option('--dataset-meta-path',
              help='Path of the MusicNet metadata (CSV format).',
              required=True)
@click.option('--fingerprints-cache-path',
              help='Path of the audio fingerprint cache.')
@click.option('--model-path',
              help='Path for the trained model.',
              required=True)
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
@click.option('--test-subsample-size',
              type=int,
              help='Absolute number of samples to use for testing.')
def main(dataset_path: Optional[str], dataset_meta_path: str,
         fingerprints_cache_path: Optional[str], model_path: Optional[str],
         column: str, model: str, split_by: str, random_state: int,
         test_split: float, test_subsample_size: int):
    dataset = MusicNet(dataset_path=dataset_path,
                       dataset_meta_path=dataset_meta_path,
                       fingerprints_cache_path=fingerprints_cache_path)
    model = joblib.load(model_path)
    _, samples_test, _, sample_labels_test = train_test_split(
        dataset=dataset,
        column=column,
        split_by=split_by,
        test_split=test_split,
        random_state=random_state)

    if test_subsample_size is not None:
        samples_test = samples_test[:test_subsample_size]
        sample_labels_test = sample_labels_test[:test_subsample_size]
    print('samples_test shape:', samples_test.shape)
    predicted_labels = np.array(
        [chunk_predict(chunk, model) for chunk in samples_test])
    diff = predicted_labels - sample_labels_test
    print('accuracy:', np.where(diff == 0).size / diff.size)


if __name__ == '__main__':
    main()
