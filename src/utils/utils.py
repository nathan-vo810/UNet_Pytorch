import random
import os


def split_train_val(img_dir, val_percent):
    '''
    :param img_dir: directory of images
    :param val_percent: validation split percentage
    :return: dictionary of train and validate index
    '''
    imgs = os.listdir(img_dir)
    imgs = [img for img in imgs if img.endswith('.jpg')]
    n_imgs = len(imgs)
    split_point = int(n_imgs * (1-val_percent))
    random.shuffle(imgs)
    dataset = {'train': imgs[:split_point], 'validate': imgs[split_point + 1:]}
    return dataset, split_point + 1


def batch(dataset, batch_size):
    batch = []
    for i, item in enumerate(dataset):
        batch.append(item)
        if (i + 1) % batch_size == 0:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch
