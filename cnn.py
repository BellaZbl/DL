'''样本不平衡cnn'''

import tensorflow as tf

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 150])
input_label = tf.placeholder(dtype=tf.int32, shape=[None])

input_y = tf.one_hot(input_label, 2, 1, 0)
input_y = tf.cast(input_y, tf.float32)

input_x = tf.reshape(input_data, shape=[-1, 5, 30])
input_x = tf.expand_dims(input_x, axis=-1)

filter_sizes = [2, 3, 5]
num_filters = 120
pooled_outputs = []

for filter_size in filter_sizes:
    conv = tf.layers.conv2d(
        input_x,
        filters=num_filters,
        kernel_size=[filter_size, 30],
        strides=(1, 1),
        padding='VALID',
        activation=tf.nn.relu
    )
    pool = tf.layers.max_pooling2d(
        conv,
        pool_size=[5 - filter_size + 1, 1],
        strides=(1, 1),
        padding='VALID'
    )
    pooled_outputs.append(pool)

h_pool = tf.concat(pooled_outputs, 3)  # concat后的shape为[句子数量，卷积后的句子长度这里是1，1，卷积核数*卷积核种类]
h_pool_flat = tf.reshape(h_pool, [-1, num_filters * len(filter_sizes)])  # reshape后的shape为[句子数量，即每个句子变成一个长度为(卷积核数*卷积核种类)的向量
h_drop = tf.nn.dropout(h_pool_flat, keep_prob=0.3)

logits = tf.layers.dense(h_drop, 2, activation=None)
prediction = tf.nn.softmax(logits)

'''coe为正负样本权重'''
coe = tf.constant([11.5, 1.0], dtype=tf.float32)
y_coe = coe * input_y

loss = tf.reduce_mean(-y_coe * tf.log(prediction + 0.000001))
optimazer = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    batch_size = 4000
    epoch_num = int(np.ceil(len(x_train) / batch_size))
    for j in range(20):
        print(j)
        for i in range(epoch_num):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end <= len(x_train):
                train_data = x_train[start:end]
                train_label = y_train[start:end]
                losses, _ = sess.run([loss, optimazer], feed_dict={input_data: train_data, input_label: train_label})
                if i % 100 == 0 and j % 3 == 0:
                    print('训练损失：', losses)
                    test_loss = sess.run(loss, feed_dict={input_data: x_test, input_label: y_test})
                    print('--------------测试损失：', test_loss)
