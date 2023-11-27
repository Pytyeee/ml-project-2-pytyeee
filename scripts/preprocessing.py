import os
import matplotlib.image as mpimg
import numpy as np
import torchvision.transforms as transforms
import torch

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

def img_resize(imgs, new_shape):
    """
    Args:
        imgs: np.array
    """
    transform = transforms.Resize(new_shape)
    resized_images = [
        transform(torch.tensor(img).float()) 
        for img in imgs
    ]

    return torch.stack(resized_images)

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

# Assign a label to a patch v
def value_to_class(patch):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(patch)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

def extract_data(filename, num_images, patching=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    data = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if i == 1 or i == num_images:
                print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            data.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(data)
    IMG_WIDTH = data[0].shape[0]
    IMG_HEIGHT = data[0].shape[1]

    if patching:
        N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

        img_patches = [
            img_crop(data[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
        ]

        data = [
            img_patches[i][j]
            for i in range(len(img_patches))
            for j in range(len(img_patches[i]))
        ]

    return np.asarray(data)

# Extract label images
def extract_labels(filename, num_images, patching=False):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if i == 1 or i == num_images:
                print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    
    if patching:
        gt_patches = [
            img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
        ]
        gt_imgs = np.asarray(
            [
                gt_patches[i][j]
                for i in range(len(gt_patches))
                for j in range(len(gt_patches[i]))
            ]
        )

    labels = np.asarray(
        [value_to_class(np.mean(gt_imgs[i])) for i in range(len(gt_imgs))]
    )

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)