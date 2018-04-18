from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import utils

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.data_utils import get_file
from keras.utils import Sequence

DENSENET121_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def SharedDenseNet(clss):

    blocks = [6, 12, 24, 16]
    n_frozen = 141
    n_clss = len(clss)

    model_list = list()

    input_shape = _obtain_input_shape(None,
                                      default_size=224,
                                      min_size=221,
                                      data_format=K.image_data_format(),
                                      require_flatten=False,
                                      weights='imagenet')
    img_input = Input(shape=input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    model_shared = transition_block(x, 0.5, name='pool4')

    for i in range(n_clss):
        x = dense_block(model_shared, blocks[3], name='conv5')

        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        n_attr = len(utils.attr[clss[i]])
        x = Dense(n_attr, activation='softmax', name='fc')(x)

        model_list.append(Model(img_input, x, name=clss[i]))

    weights_path = get_file(
        'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
        DENSENET121_WEIGHT_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')

    for i in range(n_clss):
        model_list[i].load_weights(weights_path, by_name=True)

    for layer in model_list[0].layers[:n_frozen]:
        layer.trainable = False
    # model_list[2].summary()
    return model_list


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    """
    return preprocess_input(x, data_format, mode='torch')

class DataSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # return np.array([
        #     resize(imread(file_name), (200, 200))
        #     for file_name in batch_x]), np.array(batch_y)

def data_generator(clss, input_size, batch_size):
    while True:
        pass


def train(clss):

    batch_size = 32
    input_size = 224
    epochs = 20

    n_clss = len(clss)
    n_train_samples = list()
    n_val_samples = list()
    train_generator = list()
    val_generator = list()

    for i in range(n_clss):
        _, t_n_train_samples, t_n_val_samples = utils.get_nb_dataset(clss[i])
        n_train_samples.append(t_n_train_samples)
        n_val_samples.append(t_n_val_samples)

        t_train_generator, t_val_generator = utils.data_augmentation(clss[i], input_size, batch_size)
        train_generator.append(t_train_generator)
        val_generator.append(t_val_generator)

    model_list = SharedDenseNet(clss)

    for i in range(n_clss):
        model_list[i].compile(loss='binary_crossentropy',
                              optimizer=Adam(1e-4),
                              metrics=['accuracy'])
    for i in range(4):
        for j in range(n_clss):
            history = model_list[j].fit_generator(
                train_generator[j],
                steps_per_epoch=n_train_samples[j] // batch_size,
                epochs=1,
                validation_data=val_generator[j],
                validation_steps=n_val_samples[j] // batch_size,
            )

    for i in range(n_clss):
        model_list[i].compile(loss='binary_crossentropy',
                              optimizer=Adam(1e-5),
                              metrics=['accuracy'])
    for i in range(8):
        for j in range(n_clss):
            history = model_list[j].fit_generator(
                train_generator[j],
                steps_per_epoch=n_train_samples[j] // batch_size,
                epochs=1,
                validation_data=val_generator[j],
                validation_steps=n_val_samples[j] // batch_size,
            )

    for i in range(n_clss):
        model_list[i].compile(loss='binary_crossentropy',
                              optimizer=Adam(1e-6),
                              metrics=['accuracy'])
    for i in range(2):
        for j in range(n_clss):
            history = model_list[j].fit_generator(
                train_generator[j],
                steps_per_epoch=n_train_samples[j] // batch_size,
                epochs=1,
                validation_data=val_generator[j],
                validation_steps=n_val_samples[j] // batch_size,
            )

    for i in range(n_clss):
        model_list[i].save_weights('../models/shared_{0}.h5'.format(clss[i]))
        print(model_list[i].evaluate_generator(val_generator[i], steps=n_val_samples[i] // batch_size))


if __name__ == '__main__':

    utils.conf()

    clss = ['collar_design_labels', 'neckline_design_labels', 'neck_design_labels']
    train(clss)
