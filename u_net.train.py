# -*- coding: utf-8 -*-


"""
MIT License

Copyright (c) 2017 sli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import numpy as np
import cv2
from u_net import u_net, total_loss
from data_loader import data_loader

def load_images(fn_list, batch_size):
    images = []
    labels = []
    for i in np.random.randint(len(fn_list), size=(batch_size,)):
        f = np.load(fn_list[i])
        images.append(cv2.resize(f['image'], (572, 572)))
        label = f['label'] / 256
        labels.append(np.reshape(cv2.resize(label, (572, 572)), (572, 572, 1)))
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    import time
    import tensorflow as tf

    DEVICE_STR = "gpu:1"
    BATCH_SIZE = 4
    OUTPUT_DIM = 1

    with tf.device(DEVICE_STR):
        u = u_net(DEVICE_STR, 1e-5, BATCH_SIZE, OUTPUT_DIM)
        output_shape = u.outputs.get_shape().as_list()
        logits = tf.nn.sigmoid(u.outputs, name="pred")

        y_placeholder = tf.placeholder(dtype=tf.float32, shape=output_shape)

        loss =  - tf.reduce_mean(y_placeholder * tf.log(logits) + (1 - y_placeholder) * tf.log(1 - logits))

        loss, _, reloss = total_loss("loss", loss)

        opt = tf.train.AdamOptimizer(learning_rate=1e-5)

        update_op = opt.minimize(loss)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("re loss", reloss)
    tf.summary.image("inputs", u.inputs)
    tf.summary.image("labels", y_placeholder)
    tf.summary.image("outputs", logits)

    merged = tf.summary.merge_all()

    DATA_PATH = "remote/u-net_data/"
    fn_list = [os.path.join(DATA_PATH, fn) for fn in os.listdir(DATA_PATH) if "npz"in fn] 
    print("\n".join(fn_list))
    with data_loader(n_workers=16, load_func=load_images) as data:
        i = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('log/u_net.T1', sess.graph)
            sess.run(tf.global_variables_initializer())
            while True:
                images, labels = data(fn_list, BATCH_SIZE)
                feed_dict = {
                    u.inputs: images,
                    y_placeholder: labels
                }
                summary, l, rl, _ = sess.run([merged, loss, reloss, update_op], feed_dict=feed_dict)
                if i > 0 and i % 1000 == 0:
                    saver.save(sess, 'models/u-net.%d'%i, global_step=i)
                if i % 10 == 0:
                    train_writer.add_summary(summary, i)
                    print("<%d> loss: %.6f, re loss: %.6f"%(i, l, rl))
                i += 1

