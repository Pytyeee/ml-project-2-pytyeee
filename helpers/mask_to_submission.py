#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
from datetime import datetime

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')

def make_prediction_file(test_dir, submission_dir):
    """
    Makes a submission in the SUBMISSION_DIR directory and creates it if it does not exist
    With the name:
    if you have not loaded a model: submi_date_YYYY-MM-DD_i.csv  where i is the number of the submission at this date
    if you have loaded a model:  submi_using_model_name.csv

    Args:
        None

    Returns:
        None
    """

    if not os.path.isdir(submission_dir):
        os.mkdir(submission_dir)
        print("Created the directory " + submission_dir)

    print("Making prediction file")

    date = datetime.now().strftime("%Y-%m-%d")
    submission_filename = submission_dir + "submi_date_" + date + ".csv"

    image_filenames = []
    all_dir = os.listdir(test_dir)
    all_dir.sort(key=extract_number) #we have to sort them !

    for image_name in all_dir:
        image_filename = test_dir + image_name
        image_filenames.append(image_filename)

    if test_dir + ".ipynb_checkpoints" in image_filenames:
      image_filenames.remove(test_dir + ".ipynb_checkpoints")

    print(image_filenames)

    masks_to_submission(submission_filename, *image_filenames)
    print(f"Submission file made in the location {submission_filename}")

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"(\d+)(?=\D*$)", image_filename).group(1))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    submission_filename = 'dummy_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'training/groundtruth/satImage_' + '%.3d' % i + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
