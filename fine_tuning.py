import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications import *
from keras.utils import multi_gpu_model

import utils


def train(cls):

    n_class, n_train_samples, n_val_samples = utils.get_nb_dataset(cls)

    batch_size = 32
    input_size = 224
    epochs = 20
    n_frozen = 53   # 313 53

    train_generator, val_generator = utils.data_augmentation(cls, input_size, batch_size)

    model = DenseNet121(include_top=False, weights='imagenet')
    for layer in model.layers[:n_frozen]:
        layer.trainable = False
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_class, activation='softmax')(x)
    model_final = Model(model.input, x)
    model_final.compile(loss='binary_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                        metrics=['accuracy'])

    model_final.load_weights('../models/transfer_{0}.h5'.format(cls))

    checkpointer = ModelCheckpoint(filepath='../models/transfer_{0}.h5'.format(cls), verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2, min_lr=0.000001, verbose=1)

    csv_logger = CSVLogger('../log/transfer_{0}.log'.format(cls), append=True)

    history = model_final.fit_generator(
        train_generator,
        callbacks=[EarlyStopping(patience=6), reduce_lr, checkpointer, csv_logger],
        steps_per_epoch=n_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=n_val_samples // batch_size,
    )

    # utils.plot_result(cls, history)

    r = model_final.evaluate_generator(val_generator, steps=n_val_samples // batch_size)
    print(r)

    # del model_final
    # tf.reset_default_graph()
    # K.clear_session()


def train_parallel(cls):

    def lr_schedule(epoch, lr):
        if epoch < 20:
            return 1e-4
        elif epoch < 35:
            return 1e-5
        else:
            return 1e-6

    class ParallelModel(Model):
        def __init__(self, ser_model, gpus):
            pmodel = multi_gpu_model(ser_model, gpus)
            self.__dict__.update(pmodel.__dict__)
            self._smodel = ser_model

        def __getattribute__(self, item):
            if 'load' in item or 'save' in item:
                return getattr(self._smodel, item)
            return super(ParallelModel, self).__getattribute__(item)

    n_class, n_train_samples, n_val_samples = utils.get_nb_dataset(cls)

    batch_size = 32
    input_size = 224
    epochs = 30
    n_frozen = 313

    train_generator, val_generator = utils.data_augmentation(cls, input_size, batch_size)

    with tf.device('/cpu:0'):
        model = DenseNet121(include_top=False, weights='imagenet')
        for layer in model.layers[:n_frozen]:
            layer.trainable = False
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(n_class, activation='softmax')(x)
        model_final = Model(model.input, x)

    model_final.load_weights('../models/transfer_{0}.h5'.format(cls))

    parallel_model = ParallelModel(model_final, 2)
    parallel_model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='../models/transfer_{0}.h5'.format(cls),
                                   save_weights_only=True,
                                   verbose=1,
                                   save_best_only=True)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                              patience=2, min_lr=0.00001, verbose=1)
    reduce_lr = LearningRateScheduler(lr_schedule, verbose=1)

    csv_logger = CSVLogger('../log/transfer_{0}.log'.format(cls), append=True)

    history = parallel_model.fit_generator(
        train_generator,
        callbacks=[EarlyStopping(patience=7), reduce_lr, checkpointer, csv_logger],
        steps_per_epoch=n_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=n_val_samples // batch_size,
    )

    # utils.plot_result(cls, history)

    # r = model_final.evaluate_generator(val_generator, steps=n_val_samples // batch_size)
    # print(r)


def test():
    pass


if __name__ == '__main__':

    utils.conf()

    train(utils.classes[0])

    # utils.conf()
    #
    # for i in range(1):
    #     cls = utils.classes[i]
    #     train(cls)
