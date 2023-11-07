import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications, models
import matplotlib.pyplot as plt


def getImg():
    # 测试图片
    path = 'archive/dogs-vs-cat-small/train/dogs/dog.444.jpg'
    img = image.load_img(path, target_size=(224, 224, 3))
    # 转换成数组
    img = image.img_to_array(img)
    # 升维成（1,224,224,3)
    img = np.expand_dims(img, axis=0)
    # RGB最大值为255，输入前进行标准化
    img /= 255.
    return img


def getLayerOutput(layername):
    # 使用 VGG16 模型
    vgg16 = applications.VGG16(weights='imagenet')
    # 获取层
    layer = vgg16.get_layer(layername)
    # 获取层输出
    layerout = layer.output
    # 以 vgg16 建立 model，获取输出层
    model = models.Model(inputs=vgg16.input, outputs=layerout)
    # 输入图片运算后返回层输出
    outputs = model.predict(getImg())
    return outputs


def display(layername):
    # 获取 axes
    fig, axes = plt.subplots(5, 5, figsize=(50, 50))
    # 获取层输出
    outputs = getLayerOutput(layername)
    for ax in axes.ravel():
        # 随机抽取 25 个 channel 进行显示
        high = len(outputs[0, 0, 0])
        index = np.random.randint(low=0, high=high)
        ax.imshow(outputs[0, :, :, index])
    plt.show()


if __name__ == '__main__':
    display('block1_pool')
    display('block2_pool')
    display('block3_pool')
    display('block4_pool')
    display('block5_pool')

