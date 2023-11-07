from tensorflow.keras import applications, models, layers, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_model():
    # 获取 VGG16 模型，把 include_top 设置为 False，使用自定义全连接
    conv_base = applications.VGG16(weights='imagenet', include_top=False,
                                   input_shape=(256, 256, 3))
    # 把 trainable 属性设计为 False，冻结卷积层权重
    conv_base.trainable = False
    # 新建模型，使用 VGG16 的卷积层，拉直后，自定义二层全连接层
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # 显示模型层特征
    model.summary()
    return model


def test():
    # 猫狗图路径
    path = 'archive/dogs-vs-cat-small/'
    # 训练数据集使用增强数据
    train = ImageDataGenerator(rescale=1. / 255, rotation_range=20,
                               width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.2, zoom_range=0.2,
                               horizontal_flip=True, fill_mode='nearest')
    # 验证数据集和测试数据使用原数据
    test = ImageDataGenerator(rescale=1. / 255)
    # 图片统一转换成 256*256，每批 50个
    trainData = train.flow_from_directory(path + 'train', target_size=(256, 256),
                                          batch_size=50, class_mode='binary')
    validationData = test.flow_from_directory(path + 'validation', target_size=(256, 256),
                                              batch_size=50, class_mode='binary')
    testData = test.flow_from_directory(path + 'test', target_size=(256, 256),
                                        batch_size=50, class_mode='binary')
    # 获取模型
    model = get_model()
    # 使用 adam 优化器，binary_crossentropy 二进制交叉熵损失函数
    model.compile(optimizer=optimizers.Adam(3e-4),
                  loss=losses.binary_crossentropy,
                  metrics=['acc'])
    # # 日志记录
    # callback = callbacks.TensorBoard(log_dir='logs/091902')
    # # 训练数据 2000 个，每批 50 个，所以 steps_per_epoch 训练批次为 40
    # # 验证数据 1000 个，每批 50 个，所以 validation_steps 训练批次为 20
    # # 重复训练 30 次
    # model.fit(trainData, steps_per_epoch=40, epochs=30,
    #           validation_data=validationData, validation_steps=20,
    #           callbacks=callback)

    # 训练数据 2000 个，每批 50 个，所以 steps_per_epoch 训练批次为 40
    # 验证数据 1000 个，每批 50 个，所以 validation_steps 训练批次为 20
    # 重复训练 30 次
    model.fit(trainData, steps_per_epoch=40, epochs=30,
              validation_data=validationData, validation_steps=20)

    print('---------------------------------test---------------------------------------')
    # 测试结果
    model.fit(testData, steps_per_epoch=20)


if __name__ == '__main__':
    test()
