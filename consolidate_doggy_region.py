import cv2
from glob import glob
import os
import numpy as np


def resize_image(img_pth, desired_size=1000):

    img = cv2.imread(img_pth)

    # old_size is in (height, width) format
    old_size = img.shape[:2]

    # new_size should be in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # resize image
    img = cv2.resize(img, (new_size[1], new_size[0]))

    # calculate padding
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return new_img


# define a function for horizontally
# concatenating images of different
# heights
def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] for img in img_list)

    # image resizing
    im_list_resize = [
        cv2.resize(
            img,
            (int(img.shape[1] * h_min / img.shape[0]), h_min),
            interpolation=interpolation,
        )
        for img in img_list
    ]

    # return final image
    return cv2.hconcat(im_list_resize)


# define a function for vertically
# concatenating images of different
# widths
def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1] for img in img_list)

    # resizing images
    im_list_resize = [
        cv2.resize(
            img,
            (w_min, int(img.shape[0] * w_min / img.shape[1])),
            interpolation=interpolation,
        )
        for img in img_list
    ]
    # return final image
    return cv2.vconcat(im_list_resize)


# define a function for concatenating
# images of different sizes in
# vertical and horizontal tiles
def concat_tile_resize(list_2d, interpolation=cv2.INTER_CUBIC):
    # function calling for every
    # list of images
    img_list_v = [
        hconcat_resize(list_h, interpolation=cv2.INTER_CUBIC) for list_h in list_2d
    ]

    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)


# list of image directories
dir_list = sorted(
    glob("data/masked/*"),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

# turn dir list into 2d dir list
dir_stack = [dir_list[i : i + 4] for i in range(0, len(dir_list), 4)]

# create image stack
image_stack = [[resize_image(img) for img in h_list] for h_list in dir_stack]

image = concat_tile_resize(image_stack)

# saved collate image
# cv2.imwrite("Collated Image", image)

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.imwrite("collage.png", image)
