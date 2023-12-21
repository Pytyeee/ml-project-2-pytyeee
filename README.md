# PYTyeee-RoadSegmentation
CS-433 ML Project II


## Introduction
This project aims to identify roads in a set of aerial images from Google Maps. The task boils down to a binary classification of image regions into background or road labels, which can be done by training convolutional neural network (CNN) based models to classify each pixel or patch of the input images. The UNet architecture particularly caught our attention. The original dataset was augmented because of its limited size and in order to increase the model's performance. The best F1 score achieved on AIcrowd is 0.887.


## Requirements
To run the project, the following librairies/packages are required:
- Python >= 3.10
- Numpy >= 1.23.5
- Torch >= 1.13.1
- Matplotlib >= 3.7.1
- PIL >= 9.4.0
- Imgaug >= 0.4
- Segmentation_models_pytorch > 0.3.3

They can be installed with `pip install -r requirement.txt`
 

## Project Structure
Our project repository is organized as follows : 
```
.
├── dataset                              # Original dataset
│   ├── test_set_images                     # Testing images (without groundtruth)
|   └── training                            # Training dataset (with groundtruth)
│       ├── augmented_groundtruth               # Augmented groundtruths (contains initial groundtruths)
│       ├── augmented_images                    # Augmented images (contains initial images)
│       ├── groundtruth                         # Initial groundtruth
│       └── images                              # Initial images
├── helpers                              # Helpers scripts
│   └── mask_to_submission.py                 # Helper function to create the submission file
│   └── model_handler.py                 # Helper function to save and loads the model 
├── models                               # Trained models
├── notebooks                            # Notebook for training and augmentation
│   └── augmentation.ipynb                   # Create the augmented dataset
│   └── unet_test.ipynb                      # Make the predictions from trained model
│   └── unet_train.ipynb                     # Trains the model on augmented dataset
│   └── visualisation.ipynb                  # Visualise predictions made 
├── submissions                          # Contains the csv submission files
├── scripts                              # Contains data related scripts   
│   └── augmentation.py                      # Functions to augment the dataset
│   └── preprocessing.py                     # Functions to load the images
├── run.ipynb                            # Contains the steps to run our model and create a csv file sumbmission 
└── README.md                         

```
## Training 
We trained our model on a V100 GPU on Google Colab with 16 GO of RAM.

## Run the project 
Clone the repo `git clone <repo_url> //clone the repo`\
Open the run.ipynb file in your favorite editor and run all the cells\
You have to change the REPO_DIR parameter with the path to our repo if you run that on Colab

## Authors
- Pauline Theimer-Lienhard
- Yann Ennassih
- Timo Achard
