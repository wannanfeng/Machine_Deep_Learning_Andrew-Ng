import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 保存的是训练集里面的图像数据（本训练集有209张64x64的图像），存有RGB三种颜色数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 保存的是训练集的图像对应的分类值（[0|1]，0表示不是猫，1表示是猫）

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 和训练集相同

    classes = np.array(test_dataset["list_classes"][:])  # 以bytes类型保存的两个字符串数据，数据为：[b’non-cat’,b’cat’]。
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes