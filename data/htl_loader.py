# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:htl_loader.py
    @time:2019-03-13 22:32:19
    @github:alwin114@hotmail.com
"""
import pandas as pd
from os.path import dirname, abspath, join as join_path
import numpy as np
import tensorflow.contrib.keras as kr
from collections import Counter
import random


# 加载数据
def read_file(filename):
    pd_all = pd.read_csv(filename)
    return pd_all


# 构造平衡数据
def get_balance_corpus(corpus_size, corpus_pos, corpus_neg):
    sample_size = corpus_size // 2
    pd_corpus_balance = pd.concat([
        corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size),
        corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)
    ])
    print('评论数目(总体):%d' % pd_corpus_balance.shape[0])
    print('评论数目(正向):%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
    print('评论数目(负向):%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])
    return pd_corpus_balance


file_path = join_path(dirname(__file__), 'ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv')
pd_all = read_file(file_path)
# print(pd_all[pd_all.label == 1].shape[0])

pd_positive = pd_all[pd_all.label == 1]
pd_negative = pd_all[pd_all.label == 0]
# ChnSentiCorp_htl_ba_2000 = get_balance_corpus(2000, pd_positive, pd_negative)

# ChnSentiCorp_htl_ba_2000.sample(10)

# ChnSentiCorp_htl_ba_4000 = get_balance_corpus(4000, pd_positive, pd_negative)

ChnSentiCorp_htl_ba_6000 = get_balance_corpus(6000, pd_positive, pd_negative)

corpus_all = [(row['label'], row['review']) for index, row in ChnSentiCorp_htl_ba_6000.iterrows()]
random.shuffle(corpus_all)
train_corpus = corpus_all[:5000]
test_corpus = corpus_all[5000:5500]
val_corpus = corpus_all[5500:]


def split_content_label(corpus):
    contents, labels = [], []
    for item in corpus:
        if item[1] and str(item[1]) != 'nan':
            contents.append(list(item[1]))
            labels.append(item[0])
    return contents, labels


def build_vocab(train_corpus, vocab_dir, vocab_size=5000):
    data_train, _ = split_content_label(train_corpus)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个<PAD>来将所有文本pad为同一个长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    with open(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(corpus):
    # 对输入的corpus进行排序 按照label
    categories = []
    for item in corpus:
        if item[0] not in categories:
            categories.append(item[0])
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    '将id表示的内容转换为文字'
    return ''.join(words[x] for x in content)


def process_file(corpus, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = split_content_label(corpus)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    '''生成批次数据'''
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def get_sequence_length(x_batch):
    """
        Args:
            x_batch:a batch of input_data
        Returns:
            sequence_lenghts: a list of acutal length of  every senuence_data in input_data
    """
    sequence_lengths = []
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths
