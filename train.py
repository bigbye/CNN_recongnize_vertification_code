# coding=utf-8
from utills import *
from network import *

dropout = 0.75
learning_rate = 0.001
print_stride = 10

training_steps = 5000
train_batch_size = 64
test_batch_size = 64


def train():
    net = Model(dropout, learning_rate, MAX_CAPTCHA, CHAR_SET_LEN)
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        sess.run(net.init)
        for step in range(training_steps):
            batch_x, batch_y = gen_batch(train_batch_size)
            train_feeds = {net.x: batch_x, net.y: batch_y}
            train_loss, train_op = sess.run([net.loss, net.optimizer], train_feeds)

            if step % print_stride == 0:
                batch_x, batch_y = gen_batch(test_batch_size)
                test_feeds = {net.x: batch_x, net.y: batch_y}
                test_acc = sess.run(net.accuracy, test_feeds)
                print("第%d步，训练损失：%.5f，测试准确率：%.3f" % (step, train_loss, test_acc))
                if test_acc > 0.9:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                # break


if __name__ == '__main__':
    train()
