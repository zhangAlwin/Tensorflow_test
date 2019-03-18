# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:rnn_model_attention.py
    @time:2019-03-13 22:51:39
    @github:alwin114@hotmail.com
"""
import tensorflow as tf


class TRNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    num_layers = 1  # 隐层层数
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 6000  # 词汇表达小
    attention_size = 100  # the size of attention layer

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    lr_decay = 0.9  # learning rate decay
    keep_prob = 0.5  # dropout

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.rnn()

    def rnn(self):
        def basic_rnn_cell(rnn_size):
            return tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)

        def gru_cell(rnn_size):
            return tf.contrib.rnn.GRUCell(rnn_size, state_is_tuple=True)

        # forward
        with tf.name_scope('fw_rnn'):
            fw_rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            fw_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, output_keep_prob=self.keep_prob)

        # backward
        with tf.name_scope('bw_rnn'):
            bw_rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            bw_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=self.keep_prob)

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('bi_rnn'):
            rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, inputs=embedding_inputs,
                                                            sequence_length=self.sequence_lengths, dtype=tf.float32)
        if isinstance(rnn_output, tuple):
            rnn_output = tf.concat(rnn_output, 2)

        # attention layer
        with tf.name_scope('attention'):
            input_shape = rnn_output.shape  # (batch_size,sequence_length,hidden_size)
            sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
            hidden_size = input_shape[2].value  # hidden size of the RNN layer
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([self.config.attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        # add dropout
        with tf.name_scope('dropout'):
            # attention_output shape (batch_size,hidden_size)
            self.final_output = tf.nn.dropout(attention_output, self.keep_prob)

        # fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        # calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
