from PIL import Image
import os
import numpy as np


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def get_imgs_and_labels(index_list, imgs_dir, label_dir):
    imgs = []
    labels = []
    for index in index_list:
        img_path = os.path.join(imgs_dir + index)
        imgs.append(hwc_to_chw(Image.open(img_path)))
        label_path = os.path.join(label_dir + index)
        labels.append(hwc_to_chw(Image.open(label_path)))
    return zip(imgs, labels)
