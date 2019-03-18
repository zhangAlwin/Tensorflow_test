# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:run_rnn_base.py
    @time:2019-03-18 20:56:43
    @github:alwin114@hotmail.com
"""
from __future__ import print_function
from rnn_model import *
from data.htl_loader import *
from sklearn import metrics
import os
import time
from datetime import timedelta
import tensorflow as tf
import numpy as np

vocab_dir = 'data/ChnSentiCorp_htl_all/vocab.txt'
save_dir = 'checkpoint/textcnn_decay'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
    }
    return feed_dict


def evaluate(sess, x_, y_):
    '''评估在某一数据上的准确率和损失'''
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print('Configuring TensorBoard and Saver...')
    tensorboard_dir = 'tensorboard/textrnn_attention'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")

    # 载入训练集和验证集
    start_time = time.time()
    x_train, y_train = process_file(train_corpus, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_corpus, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print('已经消耗时间...', time_dif)

    # 创建session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer.add_graph(sess.graph)

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
        flag = False
        for epoch in range(config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

                if total_batch % config.save_per_batch == 0:
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % config.print_per_batch == 0:
                    feed_dict[model.keep_prob] = 1.0
                    loss_train, acc_train = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                    loss_val, acc_val = evaluate(sess, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=sess, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                sess.run(model.optim, feed_dict=feed_dict)
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:
                break
            config.learning_rate *= config.lr_decay


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_corpus, word_to_id, cat_to_id, config.seq_length)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess=sess, save_path=save_path)

        print('Testing...')
        loss_test, acc_test = evaluate(sess, x_test, y_test)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))

        batch_size = 128
        data_len = len(x_test)
        num_batch = int((data_len - 1) / batch_size) + 1

        y_test_cls = np.argmax(y_test, 1)
        y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果

        for i in range(num_batch):  # 逐批次处理
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            feed_dict = {
                model.input_x: x_test[start_id:end_id],
                model.keep_prob: 1.0
            }
            y_pred_cls[start_id:end_id] = sess.run(model.y_pred_cls, feed_dict=feed_dict)

        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)


if __name__ == '__main__':
    print("Configuring RNN model...")
    config = TRNNConfig()
    if not os.path.exists(vocab_dir):
        build_vocab(train_corpus, vocab_dir, config.vocab_size)

    categories, cat_to_id = read_category(train_corpus)
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextRNN(config)

    train()
