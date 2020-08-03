# Plant Leaf Identification

Identification of plants through plant leaves on the basis of their shape, color and texture features using digital image processing techniques.

## Overview

Plant Leaf Identification is a system which is able to classify **32 different species of plants** on the basis of their leaves using digital image processing techniques. The images are first preprocessed and then their shape, color and texture based features are extracted from the processed image.

A dataset was created using the extracted features to train and test the model. The model used was **Support Vector Machine Classifier** and was able to classify with **90.05% accuracy**. 

## Dataset

The dataset used is [**Flavia leaves dataset**](http://flavia.sourceforge.net) which also has the breakpoints and the names mentioned for the leaves dataset

## Dependencies

* [Numpy](http://www.numpy.org)
* [Pandas](https://pandas.pydata.org)
* [OpenCV](https://opencv.org)
* [Matplotlib](https://matplotlib.org)
* [Scikit Learn](http://scikit-learn.org/)
* [Mahotas](http://mahotas.readthedocs.io/en/latest/)

It is recommended to use [Anaconda Python 3.6 distribution](https://www.anaconda.com) and using a `Jupyter Notebook`

## Instructions

* Create the following folders in the project root - 
  * `Flavia leaves dataset` : will contain Flavia dataset
  * `mobile captures` : will contain mobile captured leaf images for additional testing purposes

## Project structure

* [single_image_process_file.ipynb](single_image_process_file.ipynb) : contains exploration of preprocessing and feature extraction techniques by operating on a single image
* [background_subtract_camera_capture_leaf_file.ipynb](background_subtract_camera_capture_leaf_file.ipynb) : contains exploration of techniques to create a background subtraction function to remove background from mobile camera captured leaf images
* [classify_leaves_flavia.ipynb](Flavia%20py%20files/classify_leaves_flavia.ipynb) : uses extracted features as inputs to the model and classifies them using SVM classifier
* [preprocess_extract_dataset_flavia.ipynb](Flavia%20py%20files/preprocess_extract_dataset_flavia.ipynb) : contains create_dataset() function which performs image pre-processing and feature extraction on the dataset. The dataset is stored in `Flavia_features.csv`

## Methodology

### 1. Pre-processing

The following steps were followed for pre-processing the image:

  1. Conversion of RGB to Grayscale image
  2. Smoothing image using Guassian filter
  3. Adaptive image thresholding using Otsu's thresholding method
  4. Closing of holes using Morphological Transformation
  5. Boundary extraction using contours

### 2. Feature extraction

Variou types of leaf features were extracted from the pre-processed image which are listed as follows:

  1. *Shape based features* : physiological length,physological width, area, perimeter, aspect ratio, rectangularity, circularity
  2. *Color based features* : mean and standard deviations of R,G and B channels
  3. *Texture based features* : contrast, correlation, inverse difference moments, entropy
  
### 3. Model building and testing

  (a) [Support Vector Machine](http://scikit-learn.org/stable/modules/svm.html) Classifier was used as the model to classify the plant species <br>
  (b) Features were then scaled using [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)<br>
  (c) Also parameter tuning was done to find the appropriate hyperparameters of the model using [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
