# Image_Segmentation

Semantic Image Segmentation Project.
Images and markup are generated from the dataset: https://cvgl.stanford.edu/projects/uav_data/
# Installation

Tested on python3.6, ubuntu18.04

## Download rep

*`git clone https://github.com/Anaph/Image_Segmentation`

*`cd Image_Segmentation`

## Create virtualenv

*`virtualenv --no-site-packages --python=python3 ImageSegmentation`

*`source ImageSegmentation/bin/activate`

## Build rt-motion-detection-opencv-python

1) Install requirements

*`sudo apt install gcc make pkg-config`

2) Build

*`cd rt-motion-detection-opencv-python`

*`python3 setup.py build`

*`python3 setup.py bdist_wheel`

*`cd ..`

## Preparing the workbench

1) Install requirements

*`pip3 install -r requirements.txt`

2) Work!


# Instructions

## Folders

*`img` - Folder with project pictures

*`legacy` - Folder with old project files

*`models` - Folder with models

*`utilities` - Folder with utilities folder for this project

## Files

*`utilities/CV_image_markup.py` - Utility for dataset markup

Example use: `python3 utilities/CV_image_markup.py -i <inputvideo> -a <inputannotation> -p <outputimage> -m <outputlabel>`

You can also use `-h`

Usage example in Dataset_Markup.ipynb

*`utilities/Model_utilities.py` - Utility for creating and training a model

Usage example in Image_Segmentation.ipynb

*`Image_Segmentation.py` - The notebook in which the model is trained and predicted

*`Dataset_Markup.py` - The notebook in which the dataset markup occurs