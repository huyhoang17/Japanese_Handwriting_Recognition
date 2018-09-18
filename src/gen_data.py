"""
Script to split training and testing data
and writing path file to 2 seperate files
- train_data.csv
- test_data.csv

datasets/ETL1C_data
├── 20
├── 27
├── 28
├── 29
├── 2a
├── 2b
├── 2c
├── 2d
├── 2e
├── 2f
...
"""
import itertools
import os
from pathlib import Path

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from PIL import Image

import src.config as cf


def head(stream, n=10):
    """
    Return the first `n` elements of the stream, as plain list.
    """
    return list(itertools.islice(stream, n))


def no_label_dirs():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_dirs = os.listdir(
        os.path.join(root_dir, 'datasets/ETL1C_data')
    )
    return len(label_dirs)


def id2word():
    dirs = os.listdir(cf.BASE_LABEL_DIR)
    dirs.sort()
    return {index: char for index, char in enumerate(dirs)}


def word2id():
    return {value: key for key, value in id2word().items()}


def train_test_split_file(train_size=0.8):
    label_dirs = os.listdir(cf.BASE_LABEL_DIR)
    labels = [os.path.join(cf.BASE_LABEL_DIR, label_dir)
              for label_dir in label_dirs]
    with open(cf.TRAIN_FILE, 'w') as f_train, \
            open(cf.TEST_FILE, 'w') as f_test:
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
                    f_train.write(os.path.abspath(str(file)) + '\n')
                for file in test_test:
                    f_test.write(os.path.abspath(str(file)) + '\n')


def data_generator(filename, batch_size=16):
    """
    https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly
    """
    def gen_data():
        with open(filename) as f:
            for line in f:
                yield line.strip()

    no_dirs = no_label_dirs()
    word_to_id = word2id()
    data_iter = gen_data()
    while True:
        path_files = head(data_iter, n=batch_size)
        if len(path_files) == 0:
            break
        imgs, indexes, labels = [], [], []
        for path_file in path_files:
            img = load_img(path_file)
            img = img.resize((cf.IMAGE_SIZE, cf.IMAGE_SIZE), Image.ANTIALIAS)
            img = img_to_array(img)
            img = img / 255
            imgs.append(img)

            label = path_file.split('/')[-2]
            labels.append(label)
            index = word_to_id[label]
            indexes.append(index)

        # np.float32
        X = np.array(imgs)
        Y = np_utils.to_categorical(
            indexes,
            no_dirs
        )
        yield X, Y
