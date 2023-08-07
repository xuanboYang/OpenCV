# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
1. 更改颜色空间
"""
# 获取所有的颜色通道(以COLOR_开头的属性)
# 在OpenCV中HSV颜色空间的取值范围为：H->[0,179], S->[0,255], V->[0,255]; 其它图像处理软件不一样
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print("总颜色转换方式:{}".format(len(flags)))
print(flags)

# 转换颜色空间
# 加载数据
# img = cv.imread('./datas/xiaoren.png')
# # 将图像转换为灰度图像
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # 图像可视化
# cv.imshow('image1', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 转换颜色空间
# # 加载数据
# img = cv.imread('./datas/opencv-logo.png')
# cv.imshow('image2', img)
# # 转换为HSV格式
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# # 图像可视化
# cv.imshow('image3', hsv)  # 展示的时候，是以bgr或者gray的格式进行展示的
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 转换颜色空间
# # 加载数据
# img = cv.imread('./datas/small.png')
# # 转换为HSV格式
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# print(hsv.transpose(2, 0, 1))
# print("=" * 100)
# print(img.transpose(2, 0, 1))

# 转换颜色空间
# 加载数据
# img = cv.imread('./datas/opencv-logo.png')
# # 转换为HSV格式
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# # 定义像素点范围
# # # 蓝色的范围
# # lower = np.array([100,50,50])
# # upper = np.array([130,255,255])
# # 红色的范围
# lower = np.array([150, 50, 50])
# upper = np.array([200, 255, 255])
#
# # 在这个范围的图像像素设置为255，不在这个范围的设置为0
# mask = cv.inRange(hsv, lower, upper)
# # 进行And操作进行数据合并
# dst = cv.bitwise_and(img, img, mask=mask)
# # 图像可视化
# cv.imshow('hsv', hsv)
# cv.imshow('mask', mask)
# cv.imshow('image', img)
# cv.imshow("dest", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 转换颜色空间
# 加载数据
# img = cv.imread('./datas/opencv-logo.png')
# # 转换为HSV格式
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# # 定义像素点范围
# # # 蓝色的范围
# # lower = np.array([100,50,50])
# # upper = np.array([130,255,255])
# # 红色的范围
# lower = np.array([150, 50, 50])
# upper = np.array([200, 255, 255])
# mask1 = cv.inRange(hsv, lower, upper)
# # 绿色的范围
# lower = np.array([30, 50, 50])
# upper = np.array([80, 255, 255])
# mask2 = cv.inRange(hsv, lower, upper)
# mask = cv.add(mask1, mask2)
# # 进行And操作进行数据合并
# dst = cv.bitwise_and(img, img, mask=mask)
# # 图像可视化
# cv.imshow('hsv', hsv)
# cv.imshow('mask', mask)
# cv.imshow('image', img)
# cv.imshow("dest", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
2. 大小重置
"""
# 加载图像
# img = cv.imread("./datas/xiaoren.png")
# old_height, old_width, _ = img.shape
# print("旧图像的大小, 高度={}, 宽度:{}".format(old_height, old_width))
# new_height = int(0.8 * old_height)
# new_width = 250
# print("新图像的大小, 高度={}, 宽度:{}".format(new_height, new_width))
# dst = cv.resize(img, (new_width, new_height))
# print(dst.shape)
#
# # 图像可视化
# cv.imshow('img', img)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
3. 图像平移
"""
# 加载图像
# img = cv.imread("./datas/xiaoren.png")
# img = cv.resize(img, (250, 300))
# h, w, _ = img.shape
# # 图像可视化
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # 构建一个M
# M = np.float32([
#     [1, 0, 50],  # 水平方向往右平移50个像素
#     [0, 1, -10]  # 垂直方向往上平移10个像素
# ])
# # warpAffine计算规则：src(x,y)=dst(m11*x+m12*y+m13, m21*x+m22*y+m23)
# # x和y表示的是原始图像中的坐标
# # x和y是坐标点
# dst = cv.warpAffine(img, M, (w, h))
#
# # 图像可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # 构建一个M
# M = np.float32([
#     [0.8, -0.15, 0],
#     [-0.53, 1, 0]
# ])
# # warpAffine计算规则：src(x,y)=dst(m11*x+m12*y+m13, m21*x+m22*y+m23)
# # x和y是坐标点
# dst = cv.warpAffine(img, M, (w, h))
#
# # 图像可视化
# cv.imshow('mask', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
4. 图像旋转
"""
# 加载图像
img = cv.imread("./datas/xiaoren.png")
rows, cols, _ = img.shape
# 构建一个用于旋转的M(旋转的中心点，旋转大小，尺度)
# angle:负数表示顺时针选择
M = cv.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=-20, scale=0.5)
# M = cv.getRotationMatrix2D(center=(0, 0), angle=-20, scale=1)
print(M)
# M = cv.getRotationMatrix2D(center=(cols/2, rows/2), angle=90, scale=1)
# warpAffine计算规则：src(x,y)=dst(m11*x+m12*y+m13, m21*x+m22*y+m23)
dst1 = cv.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])

M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=0.5)
print(M)
dst2 = cv.warpAffine(dst1, M, (cols, rows), borderValue=[255, 0, 0])

h, w, _ = img.shape
dst = cv.warpAffine(img, M, (w, h))
print(dst.shape)

# 图像可视化
cv.imshow('dst1', dst1)
cv.imshow('dst2', dst2)
cv.waitKey(0)
cv.destroyAllWindows()

# 90、180、270度旋转
# 顺时针旋转90度
dst1 = cv.rotate(img, rotateCode=cv.ROTATE_90_CLOCKWISE)
# 旋转180度
dst2 = cv.rotate(img, rotateCode=cv.ROTATE_180)
# 逆时针旋转90度
dst3 = cv.rotate(img, rotateCode=cv.ROTATE_90_COUNTERCLOCKWISE)
print(dst.shape)
# 图像可视化
cv.imshow('img', img)
cv.imshow('dst1', dst1)
cv.imshow('dst2', dst2)
cv.imshow('dst3', dst3)
cv.waitKey(0)
cv.destroyAllWindows()

# 水平或者垂直翻转
dst0 = cv.flip(img, 0)  # 上下翻转
dst1 = cv.flip(img, 1)  # 左右翻转
print(dst.shape)
# 图像可视化
cv.imshow('img', img)
cv.imshow('dst0', dst0)
cv.imshow('dst1', dst1)
cv.waitKey(0)
cv.destroyAllWindows()

img = np.array(range(25)).reshape((5, 5))  # [H,W]
print(img)
print(cv.flip(img, 1))
print(img[:, ::-1])

# 图像旋转变成水平一点
img = cv.imread("car3_plat.jpg")
h, w, _ = img.shape

M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
dst = cv.warpAffine(img, M, (w + 30, w // 2), borderValue=[0, 0, 0])

cv.imshow('img', img)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('car3_plat2.png', dst)

"""
仿射变换
    在仿射变换中，原图中是平行的元素在新的图像中也是平行的元素；可以任意的给定三个点来构建
"""
