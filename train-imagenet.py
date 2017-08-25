import csv
import os
import six
import time
import cv2
import keras
import numpy as np
from collections import OrderedDict
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

class CSV_Logger(Callback):
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(CSV_Logger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            # class CustomDialect(csv.excel):
            #     delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', 'epoch time'] + self.keys, dialect='excel-tab')
            if self.append_header:
                self.writer.writeheader()

        end_time = time.time() - self.epoch_time_start
        row_dict = OrderedDict({'epoch': epoch, 'epoch time': end_time})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

def preproc(x):
    x = x[..., ::-1]
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

def set_lr(epoch):
    lr = 0.1 * (0.1 ** (epoch // 30))
    return lr

def load_val(val_dir, label_fname):
    labels = map(int, open(label_fname).readlines())
    tmp = np.zeros((len(labels), 1000))
    for i in range(len(labels)):
        tmp[i][labels[i] - 1] = 1
    labels = tmp

    val_fnames = sorted(os.listdir(val_dir))
    val_fnames = map(lambda x: os.path.join(val_dir, x), val_fnames)
    def load_img(fname):
        im = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))
        im = preproc(im.astype('float32'))
        return im
    val_imgs = map(load_img, val_fnames)
    return np.stack(val_imgs), labels

load_val(
        '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_img_val/',
        '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')

model = keras.applications.resnet50.ResNet50(weights=None)
model.compile(optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preproc)

train_gen = datagen.flow_from_directory(
        '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_img_train/',
        target_size=(224, 224),
        batch_size=64)

csv_logger = CSV_Logger('imagenet-keras.log', append=True)
callbacks = [
        keras.callbacks.LearningRateScheduler(set_lr),
        keras.callbacks.ModelCheckpoint('rn50.{epoch:02d}.h5'),
        csv_logger,
]
model.fit_generator(
        train_gen, steps_per_epoch=1281167 // 64,
        epochs=90,
        callbacks=callbacks,
        workers=10,
        max_queue_size=100)
