import numpy as np
from glob import glob
import os
import cv2
import json


def preprocess_mask(mask_path):

    # Read mask image in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert mask to binary
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Invert mask
    mask = cv2.bitwise_not(mask)

    # Normalize mask
    mask = mask / 255.0

    # Grayscale to BGR
    mask = cv2.merge((mask, mask, mask))

    # float to uint8
    mask = mask.astype(np.uint8)

    return mask


# Opening JSON file
f = open("mask.json")
data = json.load(f)

# Create directory
dir_name = "data/masked"

# Create target Directory if it doesn't exist
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


dogs = sorted(
    glob("data/raw/*"),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)
masks = sorted(
    glob("data/annotations/*"),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

for dog, mask in zip(dogs, masks):

    # print(dog, mask)
    # read image name
    image_name = dog.split("/")[-1]
    # print(image_name)

    # get image number
    image_number = int(image_name.split(".")[0])
    # print("Image number = ", image_number)

    # Get Accuracy of mask from JSON
    jac_score = data[image_number].get("Jaccard")
    print("Image Number = ", image_number, "Jaccard = ", jac_score)
    # read image and mask
    dog = cv2.imread(dog)
    mask = preprocess_mask(mask)

    # check if mask is accurate enough
    if jac_score > 0.85:

        # mask dog image
        only_dog = dog * mask

        # save only dog image in masked folder
        image_path = os.path.join(dir_name, image_name)
        # print("only dog image path", image_path)
        cv2.imwrite(image_path, only_dog)
