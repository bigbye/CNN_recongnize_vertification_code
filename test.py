# coding=utf-8
from utills import *
from network import *
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
dropout = 0.75
learning_rate = 0.001


# 完全手写一遍图，但是其实是一种笨办法，完全可以加载图
def test():
    text, image = gen_captcha_text_and_image()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, text, transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

    gray = np.mean(image, -1)
    test_image = gray / 255
    net = Model(dropout, learning_rate, MAX_CAPTCHA, CHAR_SET_LEN)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-4980")
        # text_list = sess.run(net.logits, feed_dict={net.x: [test_image]})
        # pred_text = text_list[0].tolist()
        # print("正确: {}  预测: {}".format(text, pred_text))
        predict = tf.argmax(tf.reshape(net.logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={net.x: test_image.reshape(-1, 60, 160, 1)})
        pred_text = text_list[0].tolist()  # 因为predict是个二维矩阵
        print("正确: {}  预测: {}".format(text, pred_text))


# def test2():
#     text, image = gen_captcha_text_and_image()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.text(0.1, 0.9, text, transform=ax.transAxes)
#     plt.imshow(image)
#     plt.show()
#     gray = np.mean(image, -1)
#     test_image = gray / 255
#     with tf.Session() as sess:
#         new_saver = tf.train.import_meta_graph('./model/crack_capcha.model-4990.meta')
#         new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
#         graph=tf.get_default_graph()
#         print_tensors_in_checkpoint_file("./model/crack_capcha.model-4990.meta")
#         # logits=graph.get_tensor_by_name('net.logits:0')
#         # x=graph.get_tensor_by_name('net.x:0')
#         #
#         #
#         # predict = tf.argmax(tf.reshape(logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
#         # text_list = sess.run(predict, feed_dict={x: test_image.reshape(-1, 60, 160, 1)})
#         # pred_text = text_list[0].tolist()  # 因为predict是个二维矩阵
#         # print("正确: {}  预测: {}".format(text, pred_text))

if __name__ == '__main__':
    test()
    # test2()
