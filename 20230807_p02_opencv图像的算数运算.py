# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
1. 图像进行numpy处理
"""
# img = cv.imread("./datas/xiaoren.png")
#
# img1 = img[0:300, 0:200, :]
# img2 = img[300:600, 200:400, :]
# print(img1.shape, img2.shape)
# # 图像可视化
# cv.imshow('image', img)
# cv.imshow('img1', img1)
# cv.imshow('img2', img2)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()
#
#
# img3 = 0.3*img1 + 0.7*img2
# img3 = img3/255.0
# cv.imshow('img3', img3)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()
#
# img3 = img1 - img2
# # img3 = (img3+255)/512.0
# # img3 = img3.clip(0, 255)
# img3 = img3.astype('uint8')
# cv.imshow('img3', img3)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()

"""
2. 图像基本操作
"""
# # 读取图像数据
# img = cv.imread("./datas/xiaoren.png")
# # 图像可视化
# cv.imshow('image', img)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()
#
# img2 = img[:100, 0:300]
# print(img2.shape)
# # 图像可视化
# cv.imshow('image', img2)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()
#
# # 访问图像的像素(不建议)(第一个是第多少行，第二个是第多少列)
# px = img[255, 300]
# print("位置(255,300)对应的像素为:{}".format(px))
# blue = img[250, 300, 0]
# print("位置(250,300)对应的像素的蓝色取值为:{}".format(blue))
# # 设置所有红色像素为127
# img[:, :, 2] = 127
# print("位置(250,300)对应的像素为:{}".format(img[250, 300]))
#
# # 基于Image对象获取对应的像素值
# print("位置(250,300)对应的像素的蓝色取值为:{}".format(img.item(250, 300, 0)))
# # 设置像素值
# img.itemset((250, 300, 0), 100)
# print("位置(250,300)对应的像素的新的蓝色取值为:{}".format(img.item(250, 300, 0)))

# img = cv.imread("./datas/xiaoren.png")
# cv.imshow('image1', img)
# print(img.shape)  # (600, 510, 3)
# # 图像粘贴
# box = img[0:95, 20:240]
# cv.imshow('box', box)
# print(box.shape)  # (95, 220, 3)
# box2 = img[0:95, 280:500]
# cv.imshow('box2', box2)
# print(box2.shape)  # (95, 220, 3)
# # img[0:95, 280:500] = box
# box2 = box2 * 0.7 + box * 0.3
#
# print(box2.shape)  # (95, 220, 3)
# cv.imshow('image1-1', img)
# img[0:95, 280:500] = box2
# print(type(img), img.dtype, img.shape)  # (600, 510, 3)
# # 图像可视化
# cv.imshow('image2', img)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()

# # 图像通道的分割和合并
# b, g, r = cv.split(img)
# img = cv.merge((r, g, b))  # 将原来的r当成新图像的中b，将原来的b当成新图像中的r
# # 图像可视化
# cv.imshow('image1', img)
# # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远，当键盘任意输入的时候，结束暂停
# cv.waitKey(0)
# # 释放所有资源
# cv.destroyAllWindows()

# # 添加边框
# # 读取图片
# img = cv.imread('./datas/opencv-logo.png')
#
# # 开始添加边框
# # 直接复制
# replicate = cv.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_REPLICATE)
# # 边界反射
# reflect = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT)
# # 边界反射，边界像素不保留
# reflect101 = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
# # 边界延伸循环
# wrap = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_WRAP)
# # 添加常数
# constant = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[128, 128, 128])
#
# # 可视化
# plt.subplot(231)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title("Original")
# plt.subplot(232)
# plt.imshow(cv.cvtColor(replicate, cv.COLOR_BGR2RGB))
# plt.title("Replicate")
# plt.subplot(233)
# plt.imshow(cv.cvtColor(reflect, cv.COLOR_BGR2RGB))
# plt.title("Reflect")
# plt.subplot(234)
# plt.imshow(cv.cvtColor(reflect101, cv.COLOR_BGR2RGB))
# plt.title("Reflect101")
# plt.subplot(235)
# plt.imshow(cv.cvtColor(wrap, cv.COLOR_BGR2RGB))
# plt.title("Wrap")
# plt.subplot(236)
# plt.imshow(cv.cvtColor(constant, cv.COLOR_BGR2RGB))
# plt.title("Constant")
#
# plt.show()
#
#
# # 添加边框
# # 读取图片
# img = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3, 3))
# # 开始添加边框
# # 直接复制: 直接将边缘的像素点作为新增像素点的值
# replicate = cv.copyMakeBorder(img, top=3, bottom=3, left=3, right=3, borderType=cv.BORDER_REPLICATE)
# # 边界反射
# reflect = cv.copyMakeBorder(img, 6, 3, 3, 3, cv.BORDER_REFLECT)
# # 边界反射，边界像素不保留
# reflect101 = cv.copyMakeBorder(img, 6, 3, 3, 3, cv.BORDER_REFLECT_101)
# # 边界延伸
# wrap = cv.copyMakeBorder(img, 2, 2, 6, 6, cv.BORDER_WRAP)
# # 添加常数
# constant = cv.copyMakeBorder(img, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=0)
# print(replicate)
# print(reflect)
# print(reflect101)
# print(wrap)
# print(constant)
#
# img = np.array(np.arange(0, 54)).reshape((3, 3, 6))
# print(img)
# img2 = cv.copyMakeBorder(img, top=3, bottom=3, left=3, right=3, borderType=cv.BORDER_REPLICATE)
# print(img2.shape)
# print(np.transpose(img, (2, 0, 1)))
# print(np.transpose(img2, (2, 0, 1)))

"""
3. 图像运算
"""
# 图像合并
# 加载图像
img1 = cv.imread("./datas/xiaoren.png")
img2 = cv.imread("./datas/opencv-logo.png")

# 将大小设置为相同大小
img1 = cv.resize(img1, (300, 300))
img2 = cv.resize(img2, (300, 300))

# 添加背景
# 计算公式：dst = alphe * src1 + beta * src2 + gamma
dst = cv.addWeighted(src1=img1, alpha=0.3, src2=img2, beta=1.0, gamma=0)

# 图像可视化
cv.imshow('image1', dst)
cv.waitKey(0)
cv.destroyAllWindows()

a = img1 * 0.3 + img2 * 1.0
print(np.max(a))
a = np.clip(a, a_min=0, a_max=255).astype(np.uint8)
print(a.dtype)

# 图像可视化
cv.imshow('image2', a)
cv.waitKey(0)
cv.destroyAllWindows()


img2 = cv.imread("./datas/opencv-logo.png", 1)
# 将图像转换为灰度图像
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# 二值化操作：将输入图像(灰度图像)中所有像素值大于第二个参数的全部设置为第三个参数值
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# 图像可视化
cv.imshow('image3', img2gray)
cv.waitKey(0)
cv.destroyAllWindows()

# 图像的位运算（将logo放到图像的右上角）
# 加载图像
img1 = cv.imread("./datas/xiaoren.png")
img2 = cv.imread("./datas/opencv-logo.png")

# 获取一个新数据（右上角区域数据）
rows1, cols1, _ = img1.shape
rows, cols, channels = img2.shape
start_rows = 50
end_rows = rows + 50
start_cols = cols1 - cols - 200
end_cols = cols1 - 200
roi = img1[start_rows:end_rows, start_cols:end_cols]

# 将图像转换为灰度图像
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# 将灰度图像转换为黑白图像，做一个二值化操作
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# 对图像做一个求反的操作，即255-mask
mask_inv = cv.bitwise_not(mask)

# 获取得到背景图（对应mask_inv为True的时候，进行and操作，其它位置直接设置为0）
# 在求解bitwise_and操作的时候，如果给定mask的时候，只对mask中对应为1的位置进行and操作，其它位置直接设置为0
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# 获取得到前景图
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# 前景颜色和背景颜色合并
dst = cv.add(img1_bg, img2_fg)
# dst = img1_bg + img2_fg

# 复制粘贴
img1[start_rows:end_rows, start_cols:end_cols] = dst

# 可视化
cv.imshow('res', img1)
cv.imshow('mask', mask)
cv.imshow('mask_inv', mask_inv)
cv.imshow('img1_bg', img1_bg)
cv.imshow('img2_fg', img2_fg)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()
