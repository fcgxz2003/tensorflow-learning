import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, callbacks, datasets


def get_model():
    # 生成模型
    model = models.Sequential()
    # 32个卷积核形状为 5*5，激活函数为 relu
    model.add(layers.Conv2D(filters=32, padding='same', kernel_size=(5, 5), activation='relu'))
    # 池化层，大小2*2
    model.add(layers.MaxPool2D(2, 2))
    # 64个卷积核形状为 5*5，激活函数为 relu
    model.add(layers.Conv2D(filters=64, padding='same', kernel_size=(5, 5), activation='relu'))
    # 池化层，大小2*2
    model.add(layers.MaxPool2D(2, 2))
    # 拉直数据
    model.add(layers.Flatten())
    # 多层 MLP dropout 为 0.5
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    # 输出层激活函数为 softmax
    model.add(layers.Dense(10, activation='softmax'))
    return model


def run(X, y, model, epoch=10):
    # 输入数据转换
    X, _train = convert(X, y)
    # 生成训练模型，学习率为0.003
    model.compile(optimizer=optimizers.Adam(0.003),
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # # 日志输出
    # c = callbacks.TensorBoard(log_dir='logs')
    # model.fit(X, y, batch_size=500, epochs=epoch, callbacks=c)
    model.fit(X, y, batch_size=500, epochs=epoch)
    return model


def convert(x, y):
    # 数据格式转换
    x = x.reshape(-1, 28, 28, 1)
    x = tf.convert_to_tensor(x, tf.float32)
    y = tf.convert_to_tensor(y, tf.float32)
    return x, y


if __name__ == '__main__':
    # 获取数据集
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    # 生成模型
    model = get_model()
    # 数据测试
    run(X_train, y_train, model)
    print('------------------------test-----------------------')
    run(X_test, y_test, model, 1)
