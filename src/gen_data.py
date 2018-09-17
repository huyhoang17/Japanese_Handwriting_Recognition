"""
Script to split training and testing data
and writing path file to 2 seperate files
- train_data.csv
- test_data.csv

datasets/ETL1C_data
├── ツ
├── エ
├── ハ
├── ヲ
├── ウ
├── ソ
├── ヒ
├── ネ
├── タ
├── ン
├── ロ
...
"""
import itertools
import os
from pathlib import Path

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from PIL import Image

from src.utils import rgb2gray_image

BASE_LABEL_DIR = "../datasets/ETL1C_data"  # noqa
IMAGE_SIZE = 64
TRAIN_FILE = '../datasets/train_set.csv'
TEST_FILE = '../datasets/test_set.csv'


def head(stream, n=10):
    """
    Return the first `n` elements of the stream, as plain list.
    """
    return list(itertools.islice(stream, n))


def no_label_dirs():
    label_dirs = os.listdir(BASE_LABEL_DIR)
    return len(label_dirs)


def id2word():
    dirs = os.listdir(BASE_LABEL_DIR)
    dirs.sort()
    return {index: char for index, char in enumerate(dirs)}


def word2id():
    return {value: key for key, value in id2word().items()}


def train_test_split_file(train_size=0.8):
    label_dirs = os.listdir(BASE_LABEL_DIR)
    labels = [os.path.join(BASE_LABEL_DIR, label_dir)
              for label_dir in label_dirs]
    with open(TRAIN_FILE, 'w') as f_train, \
            open(TEST_FILE, 'w') as f_test:
        for label in labels:
            files = (x for x in Path(label).iterdir() if x.is_file())
            while True:
                sub_files = head(files, 100)
                if len(sub_files) == 0:
                    break
                set_index = int(len(sub_files) * train_size)
                train_set = sub_files[:set_index]
                test_test = sub_files[set_index:]
                for file in train_set:
                    f_train.write(str(file) + '\n')
                for file in test_test:
                    f_test.write(str(file) + '\n')


def data_generator(filename, batch_size=16):
    """
    https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def gen_data():
        with open(filename) as f:
            for line in f:
                yield line.strip()

    no_dirs = no_label_dirs()

    data_iter = gen_data()
    while True:
        path_files = head(data_iter, n=batch_size)
        if len(path_files) == 0:
            break
        imgs, indexes = [], []
        for path_file in path_files:
            img = load_img(path_file)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            img = img_to_array(img)
            img = rgb2gray_image(img)
            img = img[..., np.newaxis]
            imgs.append(img)

            label = path_file.split('/')[-2]
            index = word2id()[label]
            indexes.append(index)

        X = np.array(imgs)
        Y = np_utils.to_categorical(
            np.array(indexes).astype(np.uint8),
            no_dirs
        )
        yield X, Y
