# RSNA Intracranial Hemorrhage Detection challenge

Kaggle competition [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)

Team "Mind Blowers":

* [Yuval Reina](https://www.kaggle.com/yuval6967)
* [Zahar Chikishev](https://www.kaggle.com/zaharch)

The solution consists of the following components, run consecutively

* Prepare metadata
* Training features generating neural networks
* Training shallow neural networks based on the features and metadata
   * By Yuval
   * By Zahar
* Ensembling


## Prepare metadata

`notebooks/DICOM_metadata_to_CSV.ipynb` - traverses DICOM files and extracts metadata into a dataframe. Produces three dataframes, one for the train images and two for the stage 1&2 test images.

`notebooks/Metadata.ipynb` - gets the output of the previous notebook and post-processes the collected metadata. Prepares metadata features for training, will be used as an input to Zahar's shallow NNs. Specifically, outputs two dataframes saved in `train_md.csv` and `test_md.csv` with the metadata features. 

`Production/Prepare.ipynb`is used to prepare the train.csv and test.csv for the base mosels and yuval's Sallow NN
## Features generating neural networks
The features generating training notebooks are :
* `Production/Densenet161-folds.ipynb`
* `Production/Densenet169-folds.ipynb`
* `Production/Densenet201-folds.ipynb`
* `Production/se_resnet101.ipynb`
* `Production/se_resnext101_32x4d + prepare densenet features.ipynb`
* `se_resnext101_32x4d-new_folds.ipynb`  (5 folds for se-resnext101 and se-resnet101

These notebooks also run the models on the train and test data to extract the features (4 augmented samples for train and 8 for test)

## Shallow NN by Yuval
These networks are taking features for a full series. 
* `Post Full Head Models Train.ipynb`
* `Post Full Head Models Train - new split.ipynb`
The network that was used at the end is ResModelPoll

The inference is finally done using  `prepare_ensembling.ipynb` which use the features from the base model and the Shallow NN th create prediction that are then ensembled by Zahar.

## Shallow NN by Zahar

`notebooks/Training.ipynb` - trains a shallow neural network based on the generated features and the metadata. 

5 different models are trained on 3 folds. The difference between the models is in input features. The input features for 4 different models are generated by the following backbones

* Densenet201
* Densenet161
* Densenet169
* Se-Resnext101-32x4d
* Se-resnet101

2 of the models are also trained on 5 folds:

* Se-Resnext101-32x4d
* Se-resnet101

## Ensembling

`notebooks/Ensembling.ipynb` - ensembles the results from all shallow NNs into final predictions and prepares the final submissions.
