import os
import matplotlib.image as mpimg
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
PIXEL_DEPTH = 255

def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
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

def extract_data(filename, num_images, patching=False, test=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    def extract_name(filename, i):
        if test:
            return f"{filename}test_{i}/test_{i}.png"
        else :
            return f"{filename}satImage_{'%.3d' % i}.png"

    data = []
    for i in range(1, num_images + 1):
        image_filename = extract_name(filename, i)

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

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def split_data(x, y, ratio, seed=None):
    """
    Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing.

    Args:
        x: numpy array of shape (N, D), N samples of D features.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    if seed != None:
        np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio
    # ***************************************************
    perm = np.random.permutation(len(y))
    sep = int(ratio*len(y))

    x_perm = x[perm, :]
    y_perm = y[perm]

    x_tr, x_te, y_tr, y_te = x_perm[:sep, :], x_perm[sep:, :], y_perm[:sep], y_perm[sep:]

    print(f"Data split on ratio {ratio}: TRAINING {x_tr.shape} & {y_tr.shape} " +
    f"and TEST {x_te.shape} & {y_te.shape}")

    return x_tr, x_te, y_tr, y_te

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
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