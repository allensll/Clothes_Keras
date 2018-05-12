import os
import pandas as pd
import fnmatch
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from shutil import copy2, copytree

'''
/fashionAI
    /dataset
        /downloads
            /base
            /train
            /rank
            /z_rank
    /code
'''

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


def classify(data_path, cls, attr, rate=0.9):

    file_labels = '{0}Annotations/label.csv'.format(data_path)
    df_labels = pd.read_csv(file_labels, header=None)
    df_labels = df_labels.sample(frac=1)
    df_labels.columns = ['image_id', 'class', 'label']

    df_cls = df_labels[(df_labels['class'] == cls)].copy()
    df_cls.reset_index(inplace=True)
    df_cls.drop('index', 1)

    n = len(df_cls)
    n_class = len(attr)

    for i in range(n_class):
        cur_attr = attr[i]
        df_cls_attr = df_cls[(df_cls.label.str[i] == 'y')].copy()
        df_cls_attr.reset_index(inplace=True)
        df_cls_attr.drop('index', 1)
        n_cur_attr = len(df_cls_attr)
        print(n_cur_attr)
        for j in range(n_cur_attr):
            img_id = df_cls_attr['image_id'][j]
            img_path = data_path + img_id
            if j <= n_cur_attr * rate:
                copy2(img_path, '../dataset/{0}/train/{1}'.format(cls, cur_attr))
            else:
                copy2(img_path, '../dataset/{0}/val/{1}'.format(cls, cur_attr))


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


def get_nb_test(test_path):

    return len(fnmatch.filter(os.listdir(os.path.join(test_path, 'test')), '*.jpg'))


def conf(gpu=False):

    num_cores = 4

    if gpu:
        num_GPU = 2
        num_CPU = 4
    else:
        num_CPU = 4
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
        classes=attr[cls],
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=attr[cls],
    )

    return train_generator, val_generator


def test_generator(test_path_cls, input_size, batch_size):

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    test_generator = test_datagen.flow_from_directory(
        test_path_cls,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    return test_generator


def preprocess_dataset():
    # '../dataset/downloads/base/',

    dataset_paths = ['../dataset/downloads/train/',
                     '../dataset/downloads/rank/',
                     '../dataset/downloads/z_rank/']
    if not os.path.exists('{0}Annotations'.format(dataset_paths[1])):
        os.mkdir('{0}Annotations'.format(dataset_paths[1]))
        os.mkdir('{0}Annotations'.format(dataset_paths[2]))

        rank_labels = '../dataset/downloads/fashionAI_attributes_answer_a_20180428.csv'
        z_rank_labels = '../dataset/downloads/fashionAI_attributes_answer_b_20180428.csv'
        df_rank = pd.read_csv(rank_labels, header=None)
        df_z_rank = pd.read_csv(z_rank_labels, header=None)
        df_rank.columns = ['image_id', 'class', 'label']
        df_z_rank.columns = ['image_id', 'class', 'label']

        df_z_rank = pd.concat([df_rank, df_z_rank]).drop_duplicates(subset='image_id', keep='first')
        df_z_rank = pd.concat([df_rank, df_z_rank]).drop_duplicates(subset='image_id', keep=False)

        df_rank.to_csv('{0}Annotations/label.csv'.format(dataset_paths[1]), header=None, index=False)
        df_z_rank.to_csv('{0}Annotations/label.csv'.format(dataset_paths[2]), header=None, index=False)

    for cur_cls in classes:
        os.mkdir('../dataset/{0}'.format(cur_cls))
        os.mkdir('../dataset/{0}/train'.format(cur_cls))
        os.mkdir('../dataset/{0}/val'.format(cur_cls))
        for cur_attr in attr[cur_cls]:
            os.mkdir('../dataset/{0}/train/{1}'.format(cur_cls, cur_attr))
            os.mkdir('../dataset/{0}/val/{1}'.format(cur_cls, cur_attr))

    for i in range(len(dataset_paths)):
        for cur_cls in classes:
            classify(dataset_paths[i], cur_cls, attr[cur_cls])


def preprocess_test():

    test_path = '../dataset/downloads/week-rank/Images'
    test_question = '../dataset/downloads/week-rank/Tests/question.csv'
    os.mkdir('../rank')
    os.mkdir('../rank/Images')
    copy2(test_question, '../rank/question.csv')
    for cur_cls in classes:
        copytree(os.path.join(test_path, cur_cls), '../rank/Images/{0}/test'.format(cur_cls))


if __name__ == '__main__':
    preprocess_dataset()
