import os
import torch
import datetime

MODEL_SAVE_PATH = '../model/'

def save_model(model):
    """
    Saves the model to the MODEL_SAVE_PATH directory with the current date and time

    Args:
        model: Model to save

    Returns:
        None
    """

    if not os.path.isdir(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"model_{date}_"
    i = 0
    while os.path.exists(MODEL_SAVE_PATH + filename + str(i) + ".pt"):
        i += 1

    torch.save(model, MODEL_SAVE_PATH + filename + str(i) + ".pt")

    print("Model saved to " + MODEL_SAVE_PATH + filename + str(i) + ".pt")

def load_model(name):
    """
    Load previously saved model. Warning: loads entire architecture and structure of the model

    Args:
        name: file name containing the model
    
    Returns:
        the model
    """
    path = MODEL_SAVE_PATH + name
    assert os.path.exists(path), f"Unable to find {path}"
    
    return torch.load(path)


