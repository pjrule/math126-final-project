# Classification of Classical Music Performances with Dictionary Learning and Randomized LU Decomposition
_Parker Rule, Eric Zaks, Zoe Hsieh_

This repository contains all code necessary for reproducing the results in our final project report for Tufts MATH126: Numerical Linear Algebra.

## Dependencies
* For some experiments, we used the HDF5-formatted variant of the MusicNet dataset and the accompanying metadata (`musicnet_metadata.csv`). At the start of our project, this dataset was available on John Thickstun's website [(archived)](https://web.archive.org/web/20201020123456/homes.cs.washington.edu/~thickstn/start.html), but this website is no longer available as of December 2021 (Thickstun is no longer affiliated with the University of Washington, which formerly hosted his website). A raw WAV version of the dataset, along with the CSV-formatted metadata, is [available on Zenodo](https://zenodo.org/record/5120004#.YcVV-BNKiVI), and a Dropbox link to the HDF5 variant of the dataset (~7 GB) is available upon request. Alternatively, our preprocessing pipeline could be trivially modified to use the WAV variant.

* We used the Anaconda distribution of Python 3.9. All Python dependences, including JupyterLab, can be installed via `pip` (use `pip install -r requirements.txt`).

## Notebooks
Experimental results in sections 5.1 and 5.2 were generated using largely self-contained JupyterLab notebooks.
* `Shabat et al. 2013 (Algorithm 4.1).ipynb` [(also on Google Colab)](https://colab.research.google.com/drive/1BUvOfX_FOcEXaHXXu9T-Me7QnJFljFRd?usp=sharing): contains a randomized LU implementation (see also: `models/randomized_lu.py`) and timing experiments for section 5.1.

* `Digits Classification Using Dictionaries.ipynb`: contains a randomized LU implementation (see also: `mode    ls/randomized_lu.py`) and MNIST classification experiments for section 5.2.

## Audio classification pipeline (section 5.3)
Audio classification models can be trained and evaluated using `train.py`, e.g.
```
python train.py --dataset-meta-path [path to MusicNet metadata CSV]
                --fingerprints-cache-path [path to cache file to generate (npz format)] \
                --out-path [name of model file to generate (joblib format)] \
                --dataset-path [path
                --random-state 0 \
                --test-split 0.2 \
                --train-subsample-size 100000 \
                --column ensemble \
                --model lu \
                --verbose 1 \
                --split-by recording
```

Full usage details are available via `python train.py --help` and inline documentation.

### Training many models
We trained all models in section 5.3 in parallel on the Tufts HPC cluster using `train.sh`, which generates asynchronous Slurm jobs. Model accuracy statistics are logged to `.err` files. `train.sh` takes no command-line parameters; all paths and parameters are set at the top of the file. A Python environment with the appropriate dependencies must be activated before script execution.

### Code outline
* `train.py`: CLI frontend for the training and evaluation pipeline.
* `dataset.py`: MusicNet dataset preprocessing (FFTs, label cleaning, etc.)
* `split.py`: Utility functions for splitting data into training sets (samples) and testing sets (chunks of samples).
* `models/randomized_lu.py`: NumPy/SciPy implementation of the randomized LU algorithm (Shabat et al. 2013, Algorithm 4.1).
* `models/dict_classifier.py`: Common code for the SVD and randomized LU classifiers. Implements the `scikit-learn` classifier interface.
* `models/randomized_lu_classifier.py`: Randomized LU-based dictionary learning classifier. Implements the `scikit-learn` classifier interface.
* `models/svd_classifier.py`: SVD-based dictionary learning classifier. Implements the `scikit-learn` classifier interface.
