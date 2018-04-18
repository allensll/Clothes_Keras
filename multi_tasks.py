from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications import *
from keras.utils.vis_utils import plot_model

import utils



def train(clss):

    input_clss = list()
    model_specific = list()
    n_class = list()
    n_train_samples = list()
    n_val_samples = list()

    batch_size = 32
    input_size = 224
    epochs = 50
    n_cls = len(clss)
    n_frozen = 141
    n_share = 313

    for i in range(n_cls):
        a, b, c = utils.get_nb_dataset(clss[i])
        n_class.append(a)
        n_train_samples.append(b)
        n_val_samples.append(c)

    model = DenseNet121(include_top=False, weights='imagenet')
    plot_model(model, to_file='../models/multi_task_model.png')

    for layer in model.layers[:n_frozen]:
        layer.trainable = False
    model_shared = Model(inputs=model.input, outputs=model.layers[n_share-1].output)
    inputs = Input(shape=(None, None, 512))
    x = inputs
    x = model.layers[n_share](x)



    model_sp = Model(inputs=inputs, outputs=model)
    #
    # for i in range(n_cls):
    #     model_specific = clone_model(model.layers[n_share], model_shared.output)
    #     x = model_specific.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(n_class[i], activation='softmax', name='output_{0}'.format(clss[i]))(x)
    #     model_specific[i] = Model(, x)

    for i in range(n_cls):
        input_clss.append(Input(shape=(input_size, input_size, 3), name='input_{0}'.format(clss[i])))
        x = model_shared(input_clss[i])
        model_sp_copy = clone_model(model_sp, model_shared.output)
        x = model_sp_copy(x)
        x = GlobalAveragePooling2D()(x)
        model_specific.append(Dense(n_class[i], activation='softmax', name='output_{0}'.format(clss[i]))(x))

    model_final = Model(input_clss, model_specific)

    plot_model(model_final, to_file='../models/multi_task_model.png')





if __name__ == '__main__':

    clss = ['collar_design_labels', 'neckline_design_labels', 'neck_design_labels', 'lapel_design_labels']
    train(clss)