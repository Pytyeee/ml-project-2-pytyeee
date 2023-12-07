
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import os, sys
import imageio
from PIL import Image

def data_augmentation_rotation(image, groundtruth, degree = 90):
    '''Create new image by image rotation
    Arguments : 
        - image : input image of shape (H,W) that we want to augment
        - groundtruth : segmentation map of the corresponding input image
        - degree : value of the degree of rotation
    Return :
        - augmented_image : images rotated 
        - augmented_groundtruth : associated segmentation map rotated

    '''
    # Shape of the image on which the segmentation map is placed
    # The shape of the segmentation map array is identical to the image shape
    shape_image = groundtruth.shape  # (400,400)

    # Object representing a segmentation map associated with an image
    seg_map =  SegmentationMapsOnImage(groundtruth, shape=shape_image)
    
    '''
    seq = iaa.Sequential([
                        # Augmenter to apply affine transformations to images 
                        iaa.Affine(rotate=(-degree, degree), # Rotate image by a certain degree uniformly simpled in the interval [-degree,degree]
                                   mode='symmetric') # Pads with the reflection of the vector mirrored along the edge of the array
                        ])
    '''
    seq = iaa.Affine(rotate=(-degree, degree), # Rotate image by a certain degree uniformly simpled in the interval [-degree,degree]
                    mode='symmetric') # Pads with the reflection of the vector mirrored along the edge of the array
                        
    seq._mode_segmentation_maps = 'symmetric'
    
    # Augmentation
    augmented_image, augmented_groundtruth = seq(image=image, segmentation_maps=seg_map)
    # The augmented segmentation map is converted back to an array
    augmented_groundtruth = 255 * augmented_groundtruth.get_arr().astype('uint8')

    return augmented_image, augmented_groundtruth


def data_augmentation_flip(image, groundtruth, horonzital = True):
    '''Create new image by flipping the image
    Arguments : 
        - image : input image of shape (H,W) that we want to augment
        - groundtruth : segmentation map of the corresponding input image
        - horonzital : flip horizontally the image if True, flip vertically the image if False
    Return :
        - flipped_image : images flipped 
        - flipped_groundtruth : associated segmentation map flipped
    '''
    # Shape of the image on which the segmentation map is placed
    # The shape of the segmentation map array is identical to the image shape
    shape_image = groundtruth.shape  # (400,400)

    # Object representing a segmentation map associated with an image
    seg_map =  SegmentationMapsOnImage(groundtruth, shape=shape_image)

    # Flip the image
    if horonzital :
        seq = iaa.Sequential([
                            iaa.flip.HorizontalFlip(1)  # Flip the image horizontally
                            ]) 
    else : 
        seq = iaa.Sequential([
                            iaa.flip.VerticalFlip(1)  # Flip the image vertically
                            ])
    
    # Flipping the image
    flipped_image, flipped_groundtruth = seq(image=image, segmentation_maps=seg_map)
    # The flipped segmentation map is converted back to an array
    flipped_groundtruth = 255 * flipped_groundtruth.get_arr().astype('uint8')


    return flipped_image, flipped_groundtruth

def data_augmentation_complex(image, groundtruth):
    '''Create new image by diverses modifications
    Arguments : 
        - image : input image of shape (H,W) that we want to augment
        - groundtruth : segmentation map of the corresponding input image
    Return :
        - augmented_image : image augmented 
        - augmented_groundtruth : associated segmentation map rotated

    '''
    # Shape of the image on which the segmentation map is placed
    # The shape of the segmentation map array is identical to the image shape
    shape_image = groundtruth.shape  # (400,400)

    # Object representing a segmentation map associated with an image
    seg_map =  SegmentationMapsOnImage(groundtruth, shape=shape_image)
    
    augmenter = iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-90, 90),
                            mode = 'symmetric',
                            shear=(-8, 8)
                            )
    augmenter._mode_segmentation_maps = 'symmetric'
    
    # Augmentation sequence
    seq = iaa.Sequential([
                        iaa.Fliplr(0.5), # horizontal flips with probability equals to 0.5
                        iaa.Flipud(0.5), # vertical flips with probability equals to 0.5
                        iaa.Crop(percent=(0, 0.1)), # random crops
                        # Apply affine transformations :
                        augmenter
                        ], random_order=True) # apply augmenters in random order
    
    
    
    # Augmentation
    augmented_image, augmented_groundtruth = seq(image=image, segmentation_maps=seg_map)
    # The augmented segmentation map is converted back to an array
    augmented_groundtruth = 255 * augmented_groundtruth.get_arr().astype('uint8')

    return augmented_image, augmented_groundtruth


def create_directory(directory_name):
    '''Create a new directory and verify if it doesn't already exist
    '''

    try:
        # Create a directory
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")


def save_data_augmented(augmented_image, augmented_groundtruth):
    '''Save the new data generated by augmentation
    Arguments :
        - augmented_image :
        - augmented_groundtruth : 
    '''
    
    # Creates the augmented directory if it doesn't already exist
    directory_augmented_images = "../dataset/training/training/augmented_images"
    create_directory(directory_augmented_images)
    directory_augmented_groundtruth = "../dataset/training/training/augmented_groundtruth"
    create_directory(directory_augmented_groundtruth)

    image_dir_augmented = "../dataset/training/training/augmented_images/"
    files_augmented = os.listdir(image_dir_augmented)
    a = len(files_augmented)

    image_dir = "../dataset/training/training/images/"
    files = os.listdir(image_dir)
    n = len(files)

    # Path with the name
    image_name = "../dataset/training/training/augmented_images/satImage_" + str(n+a+1).zfill(3) + ".png"
    groundtruth_name = "../dataset/training/training/augmented_groundtruth/satImage_" + str(n+a+1).zfill(3) + ".png"

    # Save the image and the groundtruth in the augmented directories
    Image.fromarray((augmented_image * 255).astype(np.uint8)).save(image_name)
    Image.fromarray((augmented_groundtruth * 255).astype(np.uint8)).save(groundtruth_name)


def data_augmented(images, groundtruths):
    '''Augment the number of input images by rotation and flip
    Arguments :
         - images : list of image
         - groundtruths : list of groundtruth
    '''

    n = len(images)

    for i in range(n):
        augmented_image, augmented_groundtruth = data_augmentation_rotation(images[i], groundtruths[i], degree = 90)
        flipped_image_h, flipped_groundtruth_h = data_augmentation_flip(images[i], groundtruths[i], horonzital = True)
        flipped_image_v, flipped_groundtruth_v = data_augmentation_flip(images[i], groundtruths[i], horonzital = False)

        # Saving the new images and groundtruths
        save_data_augmented(augmented_image, augmented_groundtruth)
        save_data_augmented(flipped_image_h, flipped_groundtruth_h)
        save_data_augmented(flipped_image_v, flipped_groundtruth_v)


def data_augmented_complex(images, groundtruths, number):
    '''Augment the number of input images by rotation and flip
    Arguments :
         - images : list of image
         - groundtruths : list of groundtruth
    '''

    n = len(images)
    for nb in range(number):
        for i in range(n):
            augmented_image, augmented_groundtruth = data_augmentation_complex(images[i], groundtruths[i])

            # Saving the new images and groundtruths
            save_data_augmented(augmented_image, augmented_groundtruth)


    
    

