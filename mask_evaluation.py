import os
import numpy as np
from numpy import asarray
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


def read_mask(path, invert=False):
    """
    Reads path of mask and return flat numpy array
    """

    # Read a grayscale image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    #  Convert grayscale image to binary
    thresh, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Invert image
    if invert:
        image = cv2.bitwise_not(image)

    # Turn image to numpy array
    image_array = asarray(image)

    # Normalize array
    image_array = image_array / 255.0

    # Flatten array
    flat_image_array = image_array.flatten()

    return flat_image_array


""" Load masks """

mask_x = sorted(
    glob("data/annotations/*"),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)
mask_y = sorted(
    glob("data/my_annotations/png/*"),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

scores = []

for x, y in tqdm(zip(mask_x, mask_y), total=len(mask_x)):

    """Image names"""
    x_name = x.split("/")[-1]

    """Read mask and return as array """
    x = read_mask(x, invert=True)
    y = read_mask(y)

    """ Calculate metrics values"""
    jac_value = jaccard_score(x, y, labels=[0, 1], average="binary")
    f1_value = f1_score(x, y, labels=[0, 1], average="binary")
    recall_value = recall_score(x, y, labels=[0, 1], average="binary")
    precision_value = precision_score(x, y, labels=[0, 1], average="binary")
    acc_value = accuracy_score(x, y)
    scores.append(
        [x_name, jac_value, f1_value, recall_value, precision_value, acc_value]
    )

""" Save results to dataframe"""
df = pd.DataFrame(
    scores, columns=["Image", "Jaccard", "F1", "Recall", "Precision", "Accuracy"]
)
print(df.head(10))

""" Turn df to json"""
df.to_json("mask.json", orient="records")
