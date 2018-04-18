from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications import *

import utils


def train(cls):

    n_class, n_train_samples, n_val_samples = utils.get_nb_dataset(cls)

    batch_size = 32
    input_size = 224
    epochs = 20
    n_frozen = 313

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

    # model_final.load_weights('./models/transfer_{0}.h5'.format(cls))

    checkpointer = ModelCheckpoint(filepath='../models/transfer_{0}.h5'.format(cls), verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.00001, verbose=1)

    history = model_final.fit_generator(
        train_generator,
        callbacks=[EarlyStopping(patience=5), reduce_lr, checkpointer],
        steps_per_epoch=n_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=n_val_samples // batch_size,
    )

    utils.plot_result(cls, history)


def test():
    pass


if __name__ == '__main__':

    utils.conf()

    for i in range(1):
        cls = utils.classes[i]
        train(cls)
