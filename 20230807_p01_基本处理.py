# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

"""
1. 图像读取
"""
print(os.getcwd())

# imread:默认返回结果是: [H,W,C]；当加载的图像是三原色图像的时候，默认返回的通道顺序是: BGR
# NOTE: 给定加载的图像路径不允许有中文，最好也不要有空格，特别是图像文件名称
img = cv.imread("datas/small.png")
img2 = cv.imread("datas/xiaoren.png")
print(f"type(img): {type(img)}, img.shape: {img.shape}")
print(f"type(img2): {type(img2)}, img2.shape: {img2.shape}")
print(f"img[:, :, :]: {img[:, :, :]}")
print(f"img[:, :, 0]: {img[:, :, 0]}")

# Gray=img[:,:,2]*0.3+img[:,:,1]*0.59+img[:,:,0] *0.11
# 从一个绿色通道转换成灰度通道 数据类型从uint8转换为float类型，但是取值范围还是[0,255]
Gray = img2[:, :, 1] * 0.89
Gray = Gray / 255  # 数据范围从[0,255]->[0,1]
# 图像可视化
cv.imshow("image_Grey", Gray)
# 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远; 当键盘任意输入的时候，结束暂停
cv.waitKey(0)
# 释放所有资源
cv.destroyAllWindows()

print("=" * 66)
"""
2. 图像基础处理
"""

# 加载图像(如果图像加载失败，那么返回的对象xiaorenImg为None)
# 第一个参数：filename，给定图片路径参数
# 第二个参数：flags，指定图像的读取方式；默认是使用BGR模型加载图像，参考：
# https://docs.opencv.org/3.4.0/d4/da8/group__imgcodecs.html#gga61d9b0126a3e57d9277ac48327799c80af660544735200cbe942eea09232eb822
# 当设置为0表示灰度图像加载，1表示加载BGR图像, 默认为1，-1表示加载alpha透明通道的图像。
xiaorenImg = cv.imread("datas/xiaoren.png", 1)
print(f"np.shape(xiaorenImg): {np.shape(xiaorenImg)}")
# 图像可视化
cv.imshow("xiaorenImg", xiaorenImg)
# 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远; 当键盘任意输入的时候，结束暂停
cv.waitKey(0)
# 释放所有资源
cv.destroyAllWindows()

# 明确给定窗口资源
cv.namedWindow('image_xiaoren', cv.WINDOW_NORMAL)
# 图像可视化
cv.imshow('image_xiaoren', xiaorenImg)
# 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
print(cv.waitKey(0))
# 释放指定窗口资源
cv.destroyWindow('image_xiaoren')

# 图像保存
# 第一个参数是图像名称，第二个参数就是图像对象
cv.imwrite('datas/yxb_t1.png', xiaorenImg)

# # 根据不同的输入进行图像的操作
# cv.imshow('image', img)
# # 等待键盘的输入（键盘上各个键对应ASCII码， http://ascii.911cha.com/）
# k = cv.waitKey(0) & 0xFF
#
# if k == 27:
#     print(k)
#     # 当输入的是ESC退出键的时候
#     cv.destroyAllWindows()
# else:
#     # 当输入的是其他键的时候
#     cv.imwrite('t2.png', img)
#     cv.destroyAllWindows()

# 读取图像将图像转换为Matplotlib可视化
# NOTE: 如果需要可视化图像，需要注意：OpenCV中是BGR图像，而Matplotlib中是RGB的图像。
yxb_t1_img = cv.imread("datas/yxb_t1.png", cv.IMREAD_GRAYSCALE)
print(f"type(yxb_t1_img): {type(yxb_t1_img)}, yxb_t1_img.shape: {yxb_t1_img.shape}")
plt.imshow(yxb_t1_img, cmap="gray")
plt.show()
# 读取图像将图像转换为Matplotlib可视化
# NOTE: 如果需要可视化图像，需要注意：OpenCV中是BGR图像，而Matplotlib中是RGB的图像。
xiaorenImg2 = cv.imread("datas/xiaoren.png", 1)
xiaorenImg2_0 = np.zeros(xiaorenImg2.shape, dtype=xiaorenImg2.dtype)
xiaorenImg2_0[:, :, 0] = xiaorenImg2[:, :, 2]
xiaorenImg2_0[:, :, 1] = xiaorenImg2[:, :, 1]
xiaorenImg2_0[:, :, 2] = xiaorenImg2[:, :, 0]
plt.imshow(xiaorenImg2_0)
plt.show()

