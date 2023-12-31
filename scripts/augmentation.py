
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import os
import matplotlib.image as mpimg
from PIL import Image

def load_image(infilename):
    """
    Loads image as numpy array from file
    """
    data = mpimg.imread(infilename)
    return data




# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    """
    Create new image where img and gt_img are concatenated
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    """
    Make a red overlay from mask predicted_img above img
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def save_data_augmented(augmented_image, augmented_groundtruth, path_img, path_msk):
    '''
    Save the new data generated by augmentation
    '''
    files_augmented = os.listdir(path_img)
    a = len(files_augmented)

    image_name = path_img + "satImage_" + str(a+1).zfill(3) + ".png"
    groundtruth_name = path_msk + "satImage_" + str(a+1).zfill(3) + ".png"

    Image.fromarray((augmented_image * 255).astype(np.uint8)).save(image_name)
    Image.fromarray(augmented_groundtruth).save(groundtruth_name)


def data_augmentation_flip(image, groundtruth, horonzital = True):
    '''Create new image by flipping the image
    Args : 
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
    
    
    seq = iaa.Affine(rotate=degree, # Rotate image by a certain degree
                     mode='symmetric') # Pads with the reflection of the vector mirrored along the edge of the array
                        
    seq._mode_segmentation_maps = 'symmetric'
    
    # Augmentation
    augmented_image, augmented_groundtruth = seq(image=image, segmentation_maps=seg_map)
    # The augmented segmentation map is converted back to an array
    augmented_groundtruth = 255 * augmented_groundtruth.get_arr().astype('uint8')

    return augmented_image, augmented_groundtruth


def data_augmentation_crop(image, groundtruth, crop = 0.1):
    '''Create new image by bluring the image 
    Arguments : 
        - image : input image of shape (H,W) that we want to augment
        - groundtruth : segmentation map of the corresponding input image
        - crop : value of the degree of croping
    Return :
        - augmented_image : images croped 
        - augmented_groundtruth : associated segmentation map croped

    '''
    # Shape of the image on which the segmentation map is placed
    # The shape of the segmentation map array is identical to the image shape
    shape_image = groundtruth.shape  # (400,400)

    # Object representing a segmentation map associated with an image
    seg_map =  SegmentationMapsOnImage(groundtruth, shape=shape_image)
    
    
    seq = iaa.Crop(percent=crop)  #cropping the images  
   
    # Augmentation
    augmented_image, augmented_groundtruth = seq(image=image, segmentation_maps=seg_map)
    # The augmented segmentation map is converted back to an array
    augmented_groundtruth = 255 * augmented_groundtruth.get_arr().astype('uint8')

    return augmented_image, augmented_groundtruth


def data_augmentation_shear(image, groundtruth, shear = 20):
    '''Create new image by shearing the image 
    Arguments : 
        - image : input image of shape (H,W) that we want to augment
        - groundtruth : segmentation map of the corresponding input image
        - crop : value of the degree of shearing
    Return :
        - augmented_image : images sheared 
        - augmented_groundtruth : associated segmentation map sheared

    '''
    # Shape of the image on which the segmentation map is placed
    # The shape of the segmentation map array is identical to the image shape
    shape_image = groundtruth.shape  # (400,400)

    # Object representing a segmentation map associated with an image
    seg_map =  SegmentationMapsOnImage(groundtruth, shape=shape_image)
    
    shear_ = shear
    seq = iaa.Affine(shear= shear_) #shearing the images  
    
    # Augmentation
    augmented_image, augmented_groundtruth = seq(image=image, segmentation_maps=seg_map)
    # The augmented segmentation map is converted back to an array
    augmented_groundtruth = 255 * augmented_groundtruth.get_arr().astype('uint8')

    return augmented_image, augmented_groundtruth

def img_float_to_uint8(img):
    """
    Convert float image into uint8
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def data_augmented(images, groundtruths, path_img, path_msk):
    '''Augment the number of input images by rotation and flip
    Arguments :
         - images : list of image
         - groundtruths : list of groundtruth
    '''

    n = len(images)

    for i in range(n):
        augmented_image, augmented_groundtruth = data_augmentation_rotation(images[i], groundtruths[i], degree = 90)
        augmented_image_, augmented_groundtruth_ = data_augmentation_rotation(images[i], groundtruths[i], degree = 45)
        flipped_image_h, flipped_groundtruth_h = data_augmentation_flip(images[i], groundtruths[i], horonzital = True)
        flipped_image_v, flipped_groundtruth_v = data_augmentation_flip(images[i], groundtruths[i], horonzital = False)
        augmented_image_shear, augmented_groundtruth_shear = data_augmentation_shear(images[i], groundtruths[i], shear = 20)
        augmented_image_crop, augmented_groundtruth_crop = data_augmentation_crop(images[i], groundtruths[i], crop = 0.3)

        # Saving the new images and groundtruths
        save_data_augmented(images[i], img_float_to_uint8(groundtruths[i]), path_img, path_msk)
        save_data_augmented(augmented_image, augmented_groundtruth, path_img, path_msk)
        save_data_augmented(flipped_image_h, flipped_groundtruth_h, path_img, path_msk)
        save_data_augmented(flipped_image_v, flipped_groundtruth_v, path_img, path_msk)
        save_data_augmented(augmented_image_, augmented_groundtruth_, path_img, path_msk)
        save_data_augmented(augmented_image_shear, augmented_groundtruth_shear, path_img, path_msk)
        save_data_augmented(augmented_image_crop, augmented_groundtruth_crop, path_img, path_msk)

