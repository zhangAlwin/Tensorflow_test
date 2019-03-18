# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:predict_rnn_attention.py
    @time:2019-03-17 11:29:03
    @github:alwin114@hotmail.com
"""
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.keras as kr
from os.path import exists, join as join_path, dirname

from rnn_model_attention import TRNNConfig, TextRNN
from data.htl_loader import *

base_dir = 'data/ChnSentiCorp_htl_all'
vocab_dir = join_path(base_dir, 'vocab.txt')

save_dir = 'checkpoint/textrnn_attention'
save_path = join_path(save_dir, 'best_validation')  # 最佳验证结果保存路径


class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category(train_corpus)
        self.words, self.wort_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)

    def predict(self, message):
        data = [self.word_to_id[x] for x in message if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    rnn_model = RnnModel()
    test_demo = ['']
    for i in test_demo:
        print(rnn_model.predict(i))
