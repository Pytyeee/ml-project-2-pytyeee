import os
import torch
from datetime import datetime
from PIL import Image

from scripts.preprocessing import *

def save_predictions(pred_imgs, dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
        print("Created the directory " + dir)

    for i in range(len(pred_imgs)):
        imageid = "prediction_" + "_%.3d" % (i+1)

        pred_img_uint8 = img_float_to_uint8(pred_imgs[i])
        Image.fromarray(pred_img_uint8).save(dir + imageid + ".png")

def save_model(model, path):
    """
    Saves the model to the MODEL_SAVE_PATH directory with the current date and time

    Args:
        model: Model to save
        path: Parent path of the model

    Returns:
        None
    """

    if not os.path.isdir(path):
        os.mkdir(path)
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"model_{date}_"
    i = 0
    while os.path.exists(path + filename + str(i) + ".pt"):
        i += 1

    torch.save(model, path + filename + str(i) + ".pt")

    print("Model saved to " + path + filename + str(i) + ".pt")

def load_model(path):
    """
    Load previously saved model. Warning: loads entire architecture and structure of the model

    Args:
        path: file path containing the model
    
    Returns:
        the model
    """
    assert os.path.exists(path), f"Unable to find {path}"
    
    return torch.load(path)


