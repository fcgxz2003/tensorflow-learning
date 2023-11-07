import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def getfilter(layername, filterindex):
    # tensorflow2.x 以上版本需要手动关闭 eager execution
    tf.compat.v1.disable_eager_execution()
    # 使用 VGG16 模型
    vgg16 = applications.vgg16.VGG16(include_top=False)
    # 根据层名称获取层
    layer = vgg16.get_layer(layername)
    # 以该层的某个过滤器输出作为 loss
    loss = K.mean(layer.output[:, :, :, filterindex])
    # 建立 loss与 vgg16 输入特征的梯度
    # 注意 gradients 返回一个列表，因此只取其第一个元素
    grads = K.gradients(loss, vgg16.input)[0]
    # grads 的更新系数，将梯度除以 L2 范数来标准化，加上 1e-4 保证分母非 0
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # 绑定输入参数 VGG16 input 值与输出参数 loss，grads
    func = K.function([vgg16.input], [loss, grads])
    # 随机生成输入图片
    image = np.random.random((1, 50, 50, 3))
    # 根据梯度上升法重复运行 50 次，将滤波器的输出值实现最大化
    for i in range(50):
        loss, grads = func(image)
        image += grads * 0.9
    return image


# 把数据转化为 RGB 格式
def display(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


if __name__ == '__main__':
    # 5行5格
    fig, axes = plt.subplots(5, 5, figsize=(50, 50))
    filter_index = 0
    for ax in axes.ravel():
        # 过滤器从0开始显示前25个
        data = getfilter('block1_conv2', filter_index)
        # 因为 display 输出为 （1，50，50，3），输出只用第一个
        a = display(data[0])
        filter_index += 1
        ax.imshow(a)
    plt.show()
