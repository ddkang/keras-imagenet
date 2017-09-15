import csv
import os
import six
import time
import cv2
import keras
import numpy as np
import pandas as pd
from collections import OrderedDict, Iterable
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

import imagenet_utils

# FIXME
def accuracy(Y_true, Y_pred):
    val_labels_fnames = \
        '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_devkit_t12/data/val_gt_wnid.txt'
    df_labels = pd.read_csv(val_labels_fnames, names=['wnid'])
    Y_true = df_labels.values
    Y_true = map(lambda x: x[0], Y_true)

    Y_pred_dec = imagenet_utils.decode_predictions(Y_pred)

    acc = 0.0
    t5 = 0.0
    for i in xrange(len(Y_true)):
        s = set(map(lambda x: x[0], Y_pred_dec[i]))
        if Y_true[i] == Y_pred_dec[i][0][0]:
            acc += 1
        if Y_true[i] in s:
            t5 += 1
    acc /= len(Y_true)
    t5 /= len(Y_true)
    print '%f,%f' % (acc, t5)


def preproc(x):
    x = x[..., ::-1]
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

def load_val(val_dir, label_fname):
    labels = map(int, open(label_fname).readlines())
    labels = np.array(labels) - 1
    '''tmp = np.zeros((len(labels), 1000))
    for i in range(len(labels)):
        tmp[i][labels[i] - 1] = 1
    labels = tmp'''

    val_fnames = sorted(os.listdir(val_dir))
    val_fnames = map(lambda x: os.path.join(val_dir, x), val_fnames)
    def load_img(fname):
        im = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))
        im = preproc(im.astype('float32'))
        return im
    val_imgs = map(load_img, val_fnames)
    return np.stack(val_imgs), labels

X_val, Y_val = load_val(
        '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_img_val/',
        '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')

for i in xrange(90):
    model = keras.models.load_model('rn50.%02d.h5' % i)
    Y_pred = model.predict(X_val)
    accuracy(Y_val, Y_pred)
    del model
