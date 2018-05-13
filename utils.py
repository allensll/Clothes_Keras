import os
import numpy as np
import pandas as pd
import fnmatch
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from shutil import copy2

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
               'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
               'pant_length_labels']

attr = {classes[0]: ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar', 'Rib Collar'],
        classes[1]: ['Invisible', 'Strapless Neck', 'Deep V Neckline', 'Straight Neck', 'V Neckline',
                     'Square Neckline', 'Off Shoulder', 'Round Neckline', 'Sweat Heart Neck',
                     'One	Shoulder Neckline'],
        classes[2]: ['Invisible', 'Short', 'Knee', 'Midi', 'Ankle', 'Floor'],
        classes[3]: ['Invisible', 'Sleeveless', 'Cup Sleeves', 'Short Sleeves', 'Elbow Sleeves', '3or4 Sleeves',
                     'Wrist Length', 'Long Sleeves', 'Extra Long Sleeves'],
        classes[4]: ['Invisible', 'Turtle Neck', 'Ruffle Semi-High Collar', 'Low Turtle Neck', 'Draped Collar'],
        classes[5]: ['Invisible', 'High Waist Length', 'Regular Length', 'Long Length', 'Micro Length',
                     'Knee Length', 'Midi Length', 'Ankle&Floor Length'],
        classes[6]: ['Invisible', 'Notched', 'Collarless', 'Shawl Collar', 'Plus Size Shawl'],
        classes[7]: ['Invisible', 'Short Pant', 'Mid Length', '3or4 Length', 'Cropped Pant', 'Full Length']
        }


def classify(cls, attr):

    if not os.path.exists('../dataset/{0}'.format(cls)):
        os.mkdir('../dataset/{0}'.format(cls))
        os.mkdir('../dataset/{0}/train'.format(cls))
        os.mkdir('../dataset/{0}/val'.format(cls))

    file_labels = '../dataset/base/Annotations/label.csv'
    df_labels = pd.read_csv(file_labels, header=None)
    df_labels.columns = ['image_id', 'class', 'label']

    df_cls = df_labels[(df_labels['class'] == cls)].copy()
    df_cls.reset_index(inplace=True)
    df_cls.drop('index', 1)

    data_path = '../dataset/base/'
    n = len(df_cls)
    n_class = len(attr)

    for i in range(n_class):
        if not os.path.exists('../dataset/{0}/train/{1}'.format(cls, attr[i])):
            os.mkdir('../dataset/{0}/train/{1}'.format(cls, attr[i]))
        if not os.path.exists('../dataset/{0}/val/{1}'.format(cls, attr[i])):
            os.mkdir('../dataset/{0}/val/{1}'.format(cls, attr[i]))

    for i in range(n):
        print(i)
        img_id = df_cls['image_id'][i]
        img_label = df_cls['label'][i]
        img_cls = attr[img_label.find('y')]
        img_path = data_path + img_id
        if i <= n * 0.9:
            copy2(img_path, '../dataset/{0}/train/{1}'.format(cls, img_cls))
        else:
            copy2(img_path, '../dataset/{0}/val/{1}'.format(cls, img_cls))


def get_nb_dataset(cls):

    train_data_dir = '../dataset/{0}/train'.format(cls)
    val_data_dir = '../dataset/{0}/val'.format(cls)
    n_class = len(os.listdir(train_data_dir))
    n_train_samples = 0
    n_val_samples = 0

    for dir in os.listdir(train_data_dir):
        n_train_samples += len(fnmatch.filter(os.listdir(os.path.join(train_data_dir, dir)), '*.jpg'))
        n_val_samples += len(fnmatch.filter(os.listdir(os.path.join(val_data_dir, dir)), '*.jpg'))

    return n_class, n_train_samples, n_val_samples


def conf(gpu=False):

    num_cores = 2

    if gpu:
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 2
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())


def plot_result(cls, history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation acc & loss')
    plt.legend()
    plt.savefig('{0}.png'.format(cls))


def data_augmentation(cls, input_size, batch_size):

    train_data_dir = '../dataset/{0}/train'.format(cls)
    val_data_dir = '../dataset/{0}/val'.format(cls)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
    )

    return train_generator, val_generator


def ensemble_avg():
    path = '../res'
    csv_list = os.listdir(path)
    if '.DS_Store' in csv_list:
        csv_list.remove('.DS_Store')

    res = list()
    for filename in csv_list:
        df = pd.read_csv(os.path.join(path, filename), header=None)
        df.columns = ['image_id', 'class', 'label']
        res.append(df)

    df_avg = res[1]
    n_sample = len(df_avg)
    n_model = len(res)
    for i in range(n_sample):
        p_f = np.array([0.0] * len(df_avg['label'][i].split(';')))
        for j in range(n_model):
            p_s = res[j]['label'][i].split(';')
            p_f += np.array([float(p) for p in p_s])
        p_f /= n_model
        p_s = ['{:5f}'.format(p) for p in p_f]
        p = ';'.join(p_s)
        df_avg['label'][i] = p

    df_avg.to_csv('../res.csv', header=None, index=False)


if __name__ == '__main__':
    pass
    # for i in range(8):
    #     cls = classes[i]
    #     # classify(cls, attr[cls])
    #     n_class, n_train_samples, n_val_samples = get_nb_dataset(cls)
    #     print([n_class, n_train_samples, n_val_samples])
