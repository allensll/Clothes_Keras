import pandas as pd
import matplotlib.pyplot as plt

import utils


def train_analysis(filelabel='../dataset/downloads/train/Annotations/label.csv', imgname='train'):
    df_train = pd.read_csv(filelabel, header=None)
    df_train.columns = ['image_id', 'class', 'label']

    dict_train = {}
    for cls in utils.classes:
        df_cur = df_train[(df_train['class'] == cls)].copy()
        df_cur.reset_index(inplace=True)
        df_cur = df_cur.drop('index', 1)
        dict_train[cls] = df_cur

    n_cls = [len(dict_train[cls]) for cls in utils.classes]
    n_m = [sum(['m' in label for label in dict_train[cls]['label']]) for cls in utils.classes]

    fig = plt.figure(figsize=(10, 6))
    p1 = plt.bar(utils.classes, n_cls, 0.6)
    p2 = plt.bar(utils.classes, n_m, 0.6)
    plt.xticks(utils.classes, rotation=-45)
    plt.legend((p1[0], p2[0]), ('number', 'm_number'))
    plt.title('train data')
    plt.savefig('../{0}1.pdf'.format(imgname))
    # plt.show()

    fig = plt.figure(figsize=(16, 8))

    for i in range(len(utils.classes)):
        cls = utils.classes[i]
        df_cur = dict_train[cls]

        n_attr = [sum([label[j] == 'y' for label in df_cur['label']]) for j in range(len(utils.attr[cls]))]
        labels = utils.attr[cls]

        plt.subplot(241 + i)
        plt.pie(n_attr, labels=labels, autopct='%1.1f%%')
        plt.title(cls)

    plt.savefig('../{0}2.pdf'.format(imgname))
    # plt.show()


def test_analysis(filerank='../rank/question.csv', imgname='test'):
    df_rank = pd.read_csv(filerank, header=None)
    df_rank.columns = ['image_id', 'class', 'label']
    print(len(df_rank))

    dict_rank = {}
    for cls in utils.classes:
        df_cur = df_rank[(df_rank['class'] == cls)].copy()
        df_cur.reset_index(inplace=True)
        df_cur = df_cur.drop('index', 1)
        dict_rank[cls] = df_cur

    n_cls = [len(dict_rank[cls]) for cls in utils.classes]

    fig = plt.figure(figsize=(10, 6))
    plt.bar(utils.classes, n_cls, 0.6)
    plt.xticks(utils.classes, rotation=-45)
    plt.title('rank data')
    plt.savefig('../{0}1.pdf'.format(imgname))
    # plt.show()


if __name__ == '__main__':
    # train_analysis()
    train_analysis('../dataset/downloads/base/Annotations/label.csv', 'base')