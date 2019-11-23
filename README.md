
# Kaggle competition:
# [RSNA Intracranial Hemorrhage Detectionhttps](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)


Team "Mind Blowers":
====================

-   [Yuval Reina](https://www.kaggle.com/yuval6967)

-   [Zahar Chikishev](https://www.kaggle.com/zaharch)

Private Leaderboard Score: 0.04732

Private Leaderboard Place: 12

General
=======

This archive holds the code and weights which was used to create and inference
the 12th place solution in “RSNA Intracranial Hemorrhage Detection” competition.

The solution consists of the following components, run consecutively

-   Prepare data and metadata

-   Training features generating neural networks

-   Training shallow neural networks based on the features and metadata

    -   By Yuval

    -   By Zahar

-   Ensembling

ARCHIVE CONTENTS
================

-   Serialized – folder containing files for serialized training and inferencing
    of base models and shallow pooled – res.

-   Production – folder, kept as reference, holds the original notebooks used to
    train the models and the submissions

-   Notebooks – folder, holds jupyter notebooks to prepare metadata, training
    and inferencing Zahar’s shallow networks, end ensembling the full solution.

Setup
=====

HARDWARE: (The following specs were used to create the original solution)
-------------------------------------------------------------------------

### For base networks and polled -res (all files in Serialized):

CPU intel i9-9920, RAM 64G, GPU Tesla V100, GPU Titan RTX.

For …. ZAHAR

SOFTWARE (python packages are detailed separately in requirements.txt):
-----------------------------------------------------------------------

### For base networks and polled -res (all files in Serialized):

OS: Ubuntu 18.04 TLS

CUDA – 10.1

For … ZAHAR

DATA SETUP
==========

1.  Download train and test data from Kaggle and update
    `./Serialized/defenitions.py` with the train and test data

2.  If you want to use our trained models, download and inflate
    [models](https://drive.google.com/file/d/1TS2alfQ0AtURLPHXtDE9LhMHnLbfipIP/view?usp=sharing)
    (for models in Serialized) put everything in one models folder and update
    `./Serialize/defenitions.py`

Data Processing
===============

Prepare data + metadata
-----------------------

`notebooks/DICOM_metadata_to_CSV.ipynb` - traverses DICOM files and extracts
metadata into a dataframe. Produces three dataframes, one for the train images
and two for the stage 1&2 test images.

`notebooks/Metadata.ipynb` - gets the output of the previous notebook and
post-processes the collected metadata. Prepares metadata features for training,
will be used as an input to Zahar's shallow NNs. Specifically, outputs two
dataframes saved in `train_md.csv` and `test_md.csv` with the metadata features.

The last section of the notebook also prepares weights for the training images.
The weights are selected to simulate the distribution to that we encounter in
the test images.

`Production/Prepare.ipynb`is used to prepare the `train.csv` and `test.csv` for the
base mosels and yuval's Sallow NN

Training Base Models 
---------------------

`./Serialized/train_base_models.ipynb` is used to train the base models using, You
should change the 2nd cell, and enter part of the name of the GPU you use, and
the name of the model to train (look at defenitions.py for a list of names).
Beware, running this notebook to completion will take a day of two.

Training Full Head models 
--------------------------

### Yuval’s shallow model - (Pooled – Res shallow model)

`./Serialized/Post Full Head Models Train .ipynb` is used to train this shallow
networks. This notebook trains all the networks. You should change the 2nd to
reflect the GPU you use.

### Shallow NN by Zahar

`notebooks/Training.ipynb` - trains a shallow neural network based on the
generated features and the metadata. All of the models are fine-tuned after a
regular training step. The fine tuning is different in that it uses weighted
random sampling, with weights defined by `notebooks/Metadata.ipynb`.

Inferencing
-----------

### Yuval’s shallow model - (Pooled – Res shallow model):

`./Serialized/prepare_ensembling.ipynb` is used for inferencing this shallow model
and prepare the results for ensembling.

Ensembling
----------

`notebooks/Ensembling.ipynb` - ensembles the results from all shallow NNs into
final predictions and prepares the final submissions.

The two final submissions are obtained by running this notebook and the
difference is the following:

**Safe submission** ensembles regular Zahar and Yuval's models.

**Risky submission** ensembles weighted Zahar's models and regular Yuval's
models, while the ensembling uses by-sample weighted log-loss with the same
weights as defined before.
