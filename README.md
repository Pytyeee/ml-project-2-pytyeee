# PYTyeee-RoadSegmentation
CS-433 ML Project II


## Introduction
This project aims to identify roads in a set of aerial images from Google Maps. The task boils down to a binary classification of image regions into background or road labels, which can be done by training convolutional neural network (CNN) based models to classify each pixel or patch of the input images. The UNet architecture particularly caught our attention. The original dataset was augmented because of its limited size and in order to increase the model's performance. The best F1 score achieved on AIcrowd is 0.887.


## Requirements
To run the project, the following librairies/packages are required:
- Python
- Numpy
- Torch
- Matplotlib
- PIL
- Datetime
- Imgaug
- Segmentation_models_pytorch

They can be installed with `pip install -r requirements.txt`
 

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
│   └── mask_to_submission.py                 # Create the augmented dataset
│   └── model_handler.py                 # Create the augmented dataset
│   └── submission_to_mask.py                 # Create the augmented dataset
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
└── README.md                         

```

## Run the project 
We run our model by using the free GPU available on Google Colab. The notebook collab_unet.ipynb can be imported on Google Collab and runned. It imports our GitHub repository. 


## Results:

