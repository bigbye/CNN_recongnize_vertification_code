# coding=utf-8
from param import *


def gen_captcha_text_and_image():
    """
    :return:验证码的label和图片
    """

    def random_captcha_text():  # 从三个列表中找4个元素生成验证码
        """
        :return:待生成的验证码元素
        """

        captcha_text = []
        for i in range(MAX_CAPTCHA):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)  # 转换成字符串
    captcha = ImageCaptcha().generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)  # 转换成numpy矩阵
    return captcha_text, captcha_image


def txt2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        if c.isdigit():
            vector[i, int(c)] = 1
        elif c.isalpha() and c.islower():
            vector[i, ord(c) - 97 + 10] = 1
        elif c.isalpha() and c.isupper():
            vector[i, ord(c) - 65 + 26 + 10] = 1
    return vector


def gen_batch(batch_size):
    batch_x = np.zeros([batch_size, 60 * 160], np.float32)
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN], np.int32)
    for i in range(batch_size):
        text, image = gen_captcha_text_and_image()
        gray = np.mean(image, -1)
        vector = txt2vec(text)
        batch_x[i, :] = gray.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = vector.reshape([-1, MAX_CAPTCHA * CHAR_SET_LEN])

    batch_x = batch_x.reshape(-1, 60, 160, 1)
    return batch_x, batch_y
#
# if __name__ == '__main__':
#     text, image = gen_captcha_text_and_image()
#
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     ax.text(0.1, 0.9, text, transform=ax.transAxes)
#     plt.imshow(image)
#     plt.show()
