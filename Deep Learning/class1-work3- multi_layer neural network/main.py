from lr_utils import *
from dnn_utils import *
from all_function import *
import warnings
from PIL import Image
import numpy as np
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset()
    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T
    x_train = x_train / 255
    x_test = x_test / 255
    # print(y_train.shape)

    layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
    parameters = muti_layer_model(x_train, y_train, layers_dims, num_iterations=2500, print_cost=True, isPlot=True)

    #判断准确率
    predictions_train = predict(x_train, y_train, parameters)  # 训练集
    predictions_test = predict(x_test, y_test, parameters)  # 测试集

    #找出预测错误的图片
    print_mislabeled_images(classes, x_test, y_test, predictions_test)


    #预测一下自己的图片是否是cat 有bug............... 故全部注释


    # my_image = "mycat.jpg"  # change this to the name of your image file
    # my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
    #
    # fname = my_image
    # image = np.array(plt.imread(fname))
    #
    # my_image = np.array(Image.fromarray(np.uint8(image)).resize(size=(64, 64))).reshape((64 * 64 * 3, 1))
    # my_predicted_image = predict(my_image, my_label_y, parameters)
    #
    # plt.imshow(image)
    # print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    #     int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")