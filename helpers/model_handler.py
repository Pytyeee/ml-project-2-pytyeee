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

    torch.save(model.state_dict(), MODEL_SAVE_PATH + filename + str(i) + ".pt")

    print("Model saved to " + MODEL_SAVE_PATH + filename + str(i) + ".pt")