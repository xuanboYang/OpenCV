# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy

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
# img = cv.imread("./datas/xiaoren.png")
# rows, cols, _ = img.shape
# # 构建一个用于旋转的M(旋转的中心点，旋转大小，尺度)
# # angle:负数表示顺时针选择
# M = cv.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=-20, scale=0.5)
# # M = cv.getRotationMatrix2D(center=(0, 0), angle=-20, scale=1)
# print(M)
# # M = cv.getRotationMatrix2D(center=(cols/2, rows/2), angle=90, scale=1)
# # warpAffine计算规则：src(x,y)=dst(m11*x+m12*y+m13, m21*x+m22*y+m23)
# dst1 = cv.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
#
# M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=0.5)
# print(M)
# dst2 = cv.warpAffine(dst1, M, (cols, rows), borderValue=[255, 0, 0])
#
# h, w, _ = img.shape
# dst = cv.warpAffine(img, M, (w, h))
# print(dst.shape)
#
# # 图像可视化
# cv.imshow('dst1', dst1)
# cv.imshow('dst2', dst2)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 90、180、270度旋转
# # 顺时针旋转90度
# dst1 = cv.rotate(img, rotateCode=cv.ROTATE_90_CLOCKWISE)
# # 旋转180度
# dst2 = cv.rotate(img, rotateCode=cv.ROTATE_180)
# # 逆时针旋转90度
# dst3 = cv.rotate(img, rotateCode=cv.ROTATE_90_COUNTERCLOCKWISE)
# print(dst.shape)
# # 图像可视化
# cv.imshow('img', img)
# cv.imshow('dst1', dst1)
# cv.imshow('dst2', dst2)
# cv.imshow('dst3', dst3)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 水平或者垂直翻转
# dst0 = cv.flip(img, 0)  # 上下翻转
# dst1 = cv.flip(img, 1)  # 左右翻转
# print(dst.shape)
# # 图像可视化
# cv.imshow('img', img)
# cv.imshow('dst0', dst0)
# cv.imshow('dst1', dst1)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# img = np.array(range(25)).reshape((5, 5))  # [H,W]
# print(img)
# print(cv.flip(img, 1))
# print(img[:, ::-1])
#
# # 图像旋转变成水平一点
# img = cv.imread("car3_plat.jpg")
# h, w, _ = img.shape
#
# M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
# dst = cv.warpAffine(img, M, (w + 30, w // 2), borderValue=[0, 0, 0])
#
# cv.imshow('img', img)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# cv.imwrite('car3_plat2.png', dst)

"""
5. 仿射变换
       在仿射变换中，原图中是平行的元素在新的图像中也是平行的元素；可以任意的给定三个点来构建
"""
# 加载图像
# img = cv.imread("./datas/xiaoren.png")
# rows, cols, _ = img.shape
#
# # 画4条直线
# cv.line(img, pt1=(0, rows // 3), pt2=(cols, rows // 3), color=(255, 0, 0), thickness=2)
# cv.line(img, pt1=(0, 2 * rows // 3), pt2=(cols, 2 * rows // 3), color=(255, 0, 0), thickness=2)
# cv.line(img, pt1=(cols // 3, 0), pt2=(cols // 3, rows), color=(255, 0, 0), thickness=2)
# cv.line(img, pt1=(2 * cols // 3, 0), pt2=(2 * cols // 3, rows), color=(255, 0, 0), thickness=2)
# print("")
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 分布在原始图像中选择三个点以及这三个点在新图像中的位置
# pts1 = np.float32([[170, 200], [350, 200], [170, 400]])
# pts2 = np.float32([[10, 50], [200, 50], [100, 300]])
# # 构建对应的M
# M = cv.getAffineTransform(pts1, pts2)
# print(M)
# # 进行转换
# dst = cv.warpAffine(img, M, (cols, rows))
#
# # 可视化画图
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('Output')
# plt.show()

# img = cv.imread("./datas/car3_plat.jpg")
# h, w, _ = img.shape
#
# M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
# print(M)
# dst1 = cv.warpAffine(img, M, (w + 30, w // 2), borderValue=[0, 0, 0])
# h, w, _ = dst1.shape
#
# # 分布在原始图像中选择三个点以及这三个点在新图像中的位置
# pts1 = np.float32([[15, 0], [245, 0], [250, 115]])
# pts2 = np.float32([[0, 0], [230, 0], [230, 115]])
# # 构建对应的M
# M = cv.getAffineTransform(pts1, pts2)
# print(M)
# # 进行转换
# dst2 = cv.warpAffine(dst1, M, (w, h))
#
# # 可视化画图
# plt.subplot(131)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(132)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.subplot(133)
# plt.imshow(cv.cvtColor(dst2, cv.COLOR_BGR2RGB))
# plt.title('dst2')
# plt.show()

"""
6. 透视转换
       实际上就是根据给定的四个点来进行转换操作，在转换过程中图像的形状不会发现变化，
       也就是原来是直线的，转换后还是直线，要求这四个点中任意三个点均不在同一线上
"""
# 加载图像
# img = cv.imread("./datas/xiaoren.png")
# rows, cols, _ = img.shape
#
# # 画两条线
# cv.line(img, pt1=(0, rows // 2), pt2=(cols, rows // 2), color=(255, 0, 0), thickness=5)
# cv.line(img, pt1=(cols // 2, 0), pt2=(cols // 2, rows), color=(255, 0, 0), thickness=5)
# print("")
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 定义四个点
# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])  # 原始图像中的点
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])  # 新图像中的点
# # M是一个3*3的矩阵
# M = cv.getPerspectiveTransform(pts1, pts2)
# print(M)
# # 计算规则：src(x,y)=dst((m11x+m12y+m13)/(m31x+m32y+m33), (m21x+m22y+m23)/(m31x+m32y+m33))
# dst = cv.warpPerspective(img, M, (300, 300))
#
# # 可视化画图
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('Output')
# plt.show()


# img = cv.imread("./datas/car3_plat.jpg")
# h, w, _ = img.shape
# # 旋转
# M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
# print(M)
# dst1 = cv.warpAffine(img, M, (w + 30, w // 2), borderValue=[0, 0, 0])
# h, w, _ = dst1.shape
#
# # 分布在原始图像中选择三个点以及这三个点在新图像中的位置
# pts1 = np.float32([[15, 0], [245, 0], [250, 115]])
# pts2 = np.float32([[0, 0], [230, 0], [230, 115]])
# # 构建对应的M
# M = cv.getAffineTransform(pts1, pts2)
# print(M)
# # 进行转换
# dst2 = cv.warpAffine(dst1, M, (w, h))
#
# # 分布在原始图像中选择四个点以及这四个点在新图像中的位置
# pts1 = np.float32([[15, 0], [245, 0], [250, 115], [34, 100]])
# pts2 = np.float32([[0, 0], [230, 0], [230, 115], [0, 115]])
# # 构建对应的M
# M = cv.getPerspectiveTransform(pts1, pts2)
# print(M)
# dst3 = cv.warpPerspective(dst1, M, (w, h))
#
# # 可视化画图
# plt.subplot(221)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(222)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.subplot(223)
# plt.imshow(cv.cvtColor(dst2, cv.COLOR_BGR2RGB))
# plt.title('dst2')
# plt.subplot(224)
# plt.imshow(cv.cvtColor(dst3, cv.COLOR_BGR2RGB))
# plt.title('dst3')
# plt.show()


# img = cv.imread("./datas/car3_plat.jpg")
# h,w,_ = img.shape
#
# # 选择
# pts1 = np.float32([[24,12], [224,87], [197,197], [0,105]]) # 原始图像中的点 --> 需要通过模型、代码进行给定
# pts2 = np.float32([[0,0], [210,0], [210,70], [0,70]]) # 我/你/开发人员希望的点坐标
# # 构建对应的M
# M = cv.getPerspectiveTransform(pts1,pts2)
# print(M)
# dst1 = cv.warpPerspective(img,M,(210,70))
#
# # 分布在原始图像中选择三个点以及这三个点在新图像中的位置
# pts1 = np.float32([[24,12], [224,87], [197,197]])
# pts2 = np.float32([[0,0], [210,0], [210,70]])
# # 构建对应的M
# M = cv.getAffineTransform(pts1,pts2)
# print(M)
# # 进行转换
# dst2 = cv.warpAffine(img,M,(210,70))
#
# # 可视化画图
# plt.subplot(131)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(132)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.subplot(133)
# plt.imshow(cv.cvtColor(dst2, cv.COLOR_BGR2RGB))
# plt.title('dst2')
# plt.show()


"""
7. 基于透视变换提取车牌区域
       暂时不考虑四个车牌坐标点的获取方式
"""
# img = cv.imread("./datas/car.jpg")
# # 选择
# pts1 = np.float32([[114, 230], [378, 226], [377, 301], [123, 307]])  # 原始图像中的点 --> 需要通过模型、代码进行给定
# pts2 = np.float32([[0, 0], [210, 0], [210, 70], [0, 70]])  # 我/你/开发人员希望的点坐标
# # 构建对应的M
# M = cv.getPerspectiveTransform(pts1, pts2)
# print(M)
# dst1 = cv.warpPerspective(img, M, (210, 70))
#
# # 可视化画图
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.show()

# img = cv.imread("datas/car2.jpg")
# # 选择
# pts1 = np.float32([[203, 300], [488, 282], [491, 363], [213, 383]])  # 原始图像中的点 --> 需要通过模型、代码进行给定
# pts2 = np.float32([[0, 0], [210, 0], [210, 70], [0, 70]])  # 我/你/开发人员希望的点坐标
# # 构建对应的M
# M = cv.getPerspectiveTransform(pts1, pts2)
# print(M)
# dst1 = cv.warpPerspective(img, M, (210, 70))
#
# # 可视化画图
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.show()

# img = cv.imread("datas/car3.jpg")
# # 选择
# pts1 = np.float32([[40, 361], [241, 441], [214, 536], [22, 450]])  # 原始图像中的点 --> 需要通过模型、代码进行给定
# pts2 = np.float32([[0, 0], [210, 0], [210, 70], [0, 70]])  # 我/你/开发人员希望的点坐标
# # 构建对应的M
# M = cv.getPerspectiveTransform(pts1, pts2)
# print(M)
# dst1 = cv.warpPerspective(img, M, (210, 70))
# # 可视化画图
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.show()

# img = cv.imread("datas/car3.jpg")
# # 选择
# pts1 = np.float32([[40, 361], [241, 441], [214, 536], [22, 450]])  # 原始图像中的点 --> 需要通过模型、代码进行给定
# pts2 = np.float32([[100, 100], [310, 100], [310, 170], [100, 170]])  # 我/你/开发人员希望的点坐标
# # 构建对应的M
# M = cv.getPerspectiveTransform(pts1, pts2)
# print(M)
# dst1 = cv.warpPerspective(img, M, (410, 270))
# # 可视化画图
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
# plt.title('dst1')
# plt.show()


"""
8. 二值化图像
"""
# 产生一个图像(从白色到黑色的递增的形式)
# img = np.arange(255, -1, -1).reshape((1, -1))
# print(img)
# for i in range(255):
#     img = np.append(img, np.arange(255, -1, -1).reshape((1, -1)), axis=0)
# img = img.astype(np.uint8)
# print(img, img.shape)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 进行普通二值化操作(第一个参数是返回的阈值，第二个参数返回的是二值化之后的图像)
# # 普通二值化操作， 将小于等于阈值thresh的设置为0，大于该值的设置为255
# ret, thresh1 = cv.threshold(src=img, thresh=127, maxval=152, type=cv.THRESH_BINARY)
# # 反转的二值化操作， 将小于等于阈值thresh的设置为255，大于该值的设置为0
# ret, thresh2 = cv.threshold(src=img, thresh=127, maxval=152, type=cv.THRESH_BINARY_INV)
# # 截断二值化操作，将小于等于阈值thresh的设置为原始值，大于该值的设置为255
# ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
# # 0二值化操作，将小于等于阈值的设置为0，大于该值的设置为原始值
# ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
# # 反转0二值化操作，将小于等于阈值的设置为原始值，大于阈值的设置为0
# ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(images[i] / 255.0, 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# 进行自适应二值化操作
# 因为二值化操作的时候需要给定一个阈值，但是实际情况下阈值不是特别好给定的。
# 所以可以基于本身的图像数据，根据当前区域的像素值获取适合的阈值对当前区域进行二值化操作
# img = cv.imread('./datas/xiaoren.png', 0)
# # 普通二值化操作
# ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# # 使用均值的方式产生当前像素点对应的阈值，
# # 使用(x,y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值
# th2 = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
#                            thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
# # 使用高斯分布的方式产生当前像素点对应的阈值
# # 使用(x,y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值，
# # 其中权重为和当前数据有关的高斯随机数
# th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv.THRESH_BINARY, 11, 2)
#
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

"""
9. 车牌图像二值化处理
"""
# img0 = cv.imread("./datas/car.jpg", 0)
# img = cv.GaussianBlur(img0, (5, 5), 0)
#
# # 普通二值化操作
# ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# # 使用均值的方式产生当前像素点对应的阈值，
# # 使用(x,y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值
# th2 = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
#                            thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
# # 使用高斯分布的方式产生当前像素点对应的阈值
# # 使用(x,y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值，
# # 其中权重为和当前数据有关的高斯随机数
# th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv.THRESH_BINARY, 11, 2)
#
# ret4, th4 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# th5 = cv.adaptiveThreshold(th4, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv.THRESH_BINARY, 11, 2)
#
# ret6, th6 = cv.threshold(cv.GaussianBlur(th3, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#
# titles = ['Original Image', 'GaussianBlur', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
#           f'OTSU:{ret4}', f'OTSU:{ret4} + Adaptive Gaussian Thresholding',
#           f'Adaptive Gaussian Thresholding + OTSU:{ret6}']
# images = [img0, img, th1, th2, th3, th4, th5, th6]
# for i in range(len(images)):
#     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

"""
10. 图像二值化 + 高斯模糊
"""
# 产生噪音数据
# img1 = np.random.uniform(low=0, high=255, size=(300,300))
# img1 = np.random.normal(150, 100, size=(300, 300))
# print(img1)
# img1 = np.clip(img1, 0, 255)
# print(img1)
# img1 = img1.astype(np.uint8)
# # 产生背景图像
# img2 = np.zeros((300, 300), dtype=np.uint8)
# img2[100:200, 100:200] = 255
# # 合并两张图像，得到一张图像
# img = cv.addWeighted(src1=img1, alpha=0.3, src2=img2, beta=0.3, gamma=0)
#
# cv.imshow('img1', img1)
# cv.imshow('img2', img2)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 进行大津法二值化操作（其实就是找一个最大基于直方图的最大差异性的阈值点）
# # 进行普通二值化操作
# ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#
# # 进行大津法二值化操作
# ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#
# # 做一个高斯转换后，再做大津法二值化操作
# # 高斯模糊的操作
# blur = cv.GaussianBlur(img, (5, 5), 0)
# ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#
# # 画图
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding(v=127)',
#           'Original Noisy Image', 'Histogram', "Otsu's Thresholding(v={})".format(ret2),
#           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding(v={})".format(ret3)]
#
# for i in range(3):
#     # 原始图
#     plt.subplot(3, 3, i * 3 + 1),
#     plt.imshow(images[i * 3], 'gray')
#     plt.title(titles[i * 3]),
#     plt.xticks([]),
#     plt.yticks([])
#
#     # 直方图
#     plt.subplot(3, 3, i * 3 + 2),
#     plt.hist(images[i * 3].ravel(), 256)
#     plt.title(titles[i * 3 + 1]),
#     #     plt.xticks([]),
#     plt.yticks([])
#
#     # 二值化后的图
#     plt.subplot(3, 3, i * 3 + 3),
#     plt.imshow(images[i * 3 + 2], 'gray')
#     plt.title(titles[i * 3 + 2]),
#     plt.xticks([]),
#     plt.yticks([])
# plt.show()


"""
11. 图像平滑/图像模糊filter操作:
        功能：降低噪音数据对应图像判断的影响
"""
# 11.1 图像模糊化的作用
# 自定义卷积操作
# 加载图像
# img1 = cv.imread('./datas/xiaoren.png')
#
# # 自定义一个kernel核
# kernel = np.ones((3, 3), np.float32) / 9
#
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# img2 = cv.filter2D(img1, -1, kernel)
#
# # 做大小缩放
# h, w, _ = img1.shape
# w = int(w * 0.5)
# h = int(h * 0.5)
# img3 = cv.resize(img1, (w, h))
# img4 = cv.resize(img2, (w, h))
#
# w = int(w * 0.2)
# h = int(h * 0.2)
# img5 = cv.resize(img3, (w, h))
# img6 = cv.resize(cv.filter2D(img4, -1, kernel), (w, h))
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(231)
# plt.title('img1')
# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
#
# plt.subplot(232)
# plt.title('img3')
# plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
#
# plt.subplot(233)
# plt.title('img5')
# plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
#
# plt.subplot(234)
# plt.title('img2')
# plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
#
# plt.subplot(235)
# plt.title('img4')
# plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
#
# plt.subplot(236)
# plt.title('img6')
# plt.imshow(cv.cvtColor(img6, cv.COLOR_BGR2RGB))
# plt.show()
#
# # plt.savefig("a.png")

# 自定义卷积操作
# img1 = cv.imread('./datas/xiaoren.png')
# img2 = cv.GaussianBlur(img1, ksize=(5, 5), sigmaX=2.0, sigmaY=2.0)
#
# plt.figure(figsize=(20, 10))
#
# plt.subplot(2, 6, 1)
# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
# plt.subplot(2, 6, 7)
# plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
#
# for i in range(5):
#     h, w, _ = img1.shape
#     w = int(w * 0.5)
#     h = int(h * 0.5)
#     g_img = cv.GaussianBlur(img2, ksize=(5, 5), sigmaX=2.0, sigmaY=2.0)
#     img1 = cv.resize(img1, (w, h))
#     img2 = cv.resize(g_img, (w, h))
#     plt.subplot(2, 6, i + 2)
#     plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
#     plt.subplot(2, 6, i + 8)
#     plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
#
# # plt.show()
# plt.savefig("a.png")


# 11.2 自定义卷积核
# 自定义卷积操作
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
#
# # 自定义一个kernel核
# kernel = np.ones((5, 5), np.float32) / 25
#
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# dst = cv.filter2D(img, -1, kernel)
#
# plt.figure(figsize=(20, 10))
#
# # 可视化
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('Averaging')
# plt.show()

# 自定义卷积操作
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
#
# # 自定义一个kernel核
# kernel = np.array([
#     [1, 2, 1],
#     [0, -8, 0],
#     [1, 2, 1]
# ], dtype=np.float32)
#
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# dst = cv.filter2D(img, -1, kernel)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('Define Kernel')
# plt.show()

# 自定义卷积操作
# 加载图像
# img = cv.imread('./datas/xiaoren.png', 0)
#
# # 自定义一个kernel核
# kernel = np.array([
#     [1, 2, 1],
#     [0, -8, 0],
#     [1, 2, 1]
# ], dtype=np.float32)
#
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# dst = cv.filter2D(img, -1, kernel)
#
# # 自定义一个kernel核
# kernel = np.array([
#     [0, 0, 0],
#     [0, -8, 4],
#     [0, 4, 0]
# ], dtype=np.float32)
#
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# dst2 = cv.filter2D(img, -1, kernel)
#
# plt.figure(figsize=(20, 30))
# # 可视化
# plt.subplot(311)
# plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
# plt.title('Original')
#
# plt.subplot(312)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_GRAY2RGB))
# plt.title('Define Kernel')
#
# plt.subplot(313)
# plt.imshow(cv.cvtColor(dst2, cv.COLOR_GRAY2RGB))
# plt.title('Define Kernel2')
# plt.show()


"""
12. 均值滤波
    选择窗口中的所有值的均值作为输出值
"""
# OpenCV自带的均值filter
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
#
# # 做一个卷积操作
# dst = cv.blur(img, ksize=(11, 11))
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('Averaging')
# plt.show()


"""
13. 高斯滤波
    窗口滤波过程中对应的卷积参数是通过高斯函数计算出来的，
    特点是中间区域的权重系数大，而周边区域的权重系数小。 作用：去除高斯噪声数据以及图像模糊化操作
"""
# 查看5*5的高斯卷积kernel
# 窗口大小5*5，标准差为1
# a = cv.getGaussianKernel(5, 1, ktype=cv.CV_64F)
# print(a)
# kernel = np.dot(a, a.T)
# print(kernel)
#
# # OpenCV自带的高斯filter过滤器
# # 加载图像
# img = cv.imread('./datas/koala.png')
#
# # 做一个卷积操作
# # ksize：给定窗口大小
# # sigmaX: 给定横向的kernel中，参数的标准差
# # sigmaY: 给定纵向的kernel中，参数的标准差
# dst = cv.GaussianBlur(img, ksize=(9, 9), sigmaX=2, sigmaY=2)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('GaussianBlur1')
# plt.show()
#
# # 定义一个卷积核
# # CV_64F给定数据类型为浮点型64位的，也就是double这个类型
# kernel1 = cv.getGaussianKernel(9, 2, ktype=cv.CV_64F)
# kernel2 = np.transpose(kernel1)
#
# # 加载图像
# img = cv.imread('./datas/koala.png')
#
# # 做一个卷积操作
# dst = cv.filter2D(cv.filter2D(img, -1, kernel1), -1, kernel2)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('GaussianBlur1')
# plt.show()


"""
14. 中值滤波
    选择窗口区域中的中值作为输出值
"""
# 中值过滤
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
# noisy_img = np.random.normal(10, 10, (img.shape[0], img.shape[1], img.shape[2]))
# noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
# img = img + noisy_img
#
# # 转换为灰度图像
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # 做一个中值过滤
# dst = cv.medianBlur(img, ksize=5)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(121)
# plt.imshow(img, 'gray')
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(dst, 'gray')
# plt.title('medianBlur')
# plt.show()


"""
15. 双边滤波
    提取图像的纹理、条纹信息
"""
# 双边滤波: 中间的纹理删除，保留边缘信息
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # 做一个双边滤波
# dst = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('bilateralFilter')
# plt.show()


"""
15. 形态学转换
    主要包括腐蚀、扩张、打开、关闭等操作；主要操作是基于kernel核的操作
    常见的核主要有：矩阵、十字架、椭圆结构的kernel
    形态学转换见:http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm
"""
kernel1 = cv.getStructuringElement(cv.MORPH_RECT, ksize=(5, 5))
print("矩形kernel:\n{}".format(kernel1))

kernel2 = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(5, 5))
print("十字架kernel:\n{}".format(kernel2))

kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
print("椭圆kernel:\n{}".format(kernel3))
print(type(kernel3))

"""
15.1: 腐蚀
        腐蚀的意思是将边缘的像素点进行一些去除的操作；腐蚀的操作过程就是让kernel核在图像上进行滑动，
        当内核中的所有像素被视为1时，原始图像中对应位置的像素设置为1，否则设置为0.
        其主要效果是：可以在图像中减少前景图像(白色区域)的厚度，
        有助于减少白色噪音，可以用于分离两个连接的对象 一般应用与只有黑白像素的灰度情况
"""
# img = cv.imread("./datas/j.png", 0)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 进行腐蚀操作
# # a. 定义一个核(全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑)
# # 核的定义和卷积不一样，卷积里面是参数的意思，膨胀里面是范围的意思(1表示包含，0表示不考虑)
# kernel = np.ones((5, 5), np.uint8)
# # b. 腐蚀操作
# dst = cv.erode(img, kernel, iterations=1, borderType=cv.BORDER_REFLECT)
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 进行腐蚀操作
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# # kernel = cv.getStructuringElement(cv.MORPH_ERODE, (5,5))
# kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(5, 5))
# # b. 腐蚀操作
# dst = cv.morphologyEx(img, cv.MORPH_ERODE, kernel)
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
15.2: 扩张/膨胀
        和腐蚀的操作相反，其功能是增加图像的白色区域的值，
        只要在kernel中所有像素中有可以视为1的像素值，那么就将原始图像中对应位置的像素值设置为1，否则设置为0。
        通常情况下，在去除噪音后，可以通过扩张在恢复图像的目标区域信息。
"""
# 加载图像
# img = cv.imread('./datas/j.png', 0)
# # 进行扩张/膨胀操作
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. 膨胀操作
# dst = cv.dilate(img, kernel, iterations=1)
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 进行扩张/膨胀操作
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = cv.getStructuringElement(cv.MORPH_DILATE, (5, 5))
# # b. 膨胀操作
# dst = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
15.3: Open
        Open其实指的就是先做一次腐蚀，然后再做一次扩张操作，一般用于去除噪音数据。
"""
# 加载图像
# img = cv.imread('./datas/j.png', 0)
#
# # 加载噪音数据
# rows, cols = img.shape
# for i in range(100):
#     x = np.random.randint(cols)
#     y = np.random.randint(rows)
#     img[y, x] = 255
# # c. 可视化
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. Open操作
# dst = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. 先腐蚀
# dst1 = cv.erode(img, kernel, iterations=1)
# # c. 再膨胀
# dst2 = cv.dilate(dst1, kernel, iterations=1)
# # d. 可视化
# cv.imshow('img', img)
# cv.imshow('dst1', dst1)
# cv.imshow('dst2', dst2)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
15.4: Closing
        Closing其实指的就是先做一次扩张，再做一次腐蚀；对前景图像中的如果包含黑色点，有一共去除的效果。
"""
# 加载图像
# img = cv.imread('./datas/j.png', 0)
#
# # 加载噪音数据
# rows, cols = img.shape
# # 加白色点
# for i in range(100):
#     x = np.random.randint(cols)
#     y = np.random.randint(rows)
#     img[y, x] = 255
#
# # 加黑色点
# for i in range(1000):
#     x = np.random.randint(cols)
#     y = np.random.randint(rows)
#     img[y, x] = 0
# # c. 可视化
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. Closing操作
# dst = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)
#
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. 再膨胀
# dst1 = cv.dilate(img, kernel, iterations=1)
# # c. 先腐蚀
# dst2 = cv.erode(dst1, kernel, iterations=1)
#
# # d. 可视化
# cv.imshow('img', img)
# cv.imshow('dst1', dst1)
# cv.imshow('dst2', dst2)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. 先Closing操作，在Open操作
# dst1 = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)
# dst2 = cv.morphologyEx(dst1, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
#
# # c. 可视化
# cv.imshow('img', img)
# cv.imshow('dst1', dst1)
# cv.imshow('dst2', dst2)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
15.5: Morphological Gradient 形态梯度
        就是在膨胀和腐蚀之间的操作，也就是在膨胀的图像和腐蚀的图像之间取差集，
        一般的结果就是边缘位置显示，其他位置不显示；一般做这个之前，先做一个噪音数据去除的操作。
"""
# 加载图像
# img = cv.imread('./datas/j.png', 0)
# # a. 定义一个核（全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((5, 5), np.uint8)
# # b. 形态梯度(dilate - erode)
# dst = cv.morphologyEx(img, op=cv.MORPH_GRADIENT, kernel=kernel, iterations=1)
#
# # c. 可视化
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
15.6: Top Hat
        在原始图像和Open图像之间获取差集
"""
# 加载图像
# img = cv.imread('./datas/j.png', 0)
# # a. 定义一个核（全部设置为1表示对核中9*9区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((9, 9), np.uint8)
#
# # b1. Open(先腐蚀，再扩展)
# dst1 = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
# # b2. Top Hat(src - open)
# dst2 = cv.morphologyEx(img, op=cv.MORPH_TOPHAT, kernel=kernel, iterations=1)
# # dst2 = img - dst1
#
#
# # c. 可视化
# # 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(132)
# plt.imshow(dst1, 'gray')
# plt.title('Open')
#
# plt.subplot(133)
# plt.imshow(dst2, 'gray')
# plt.title('TopHat')
# plt.show()

"""
15.7: Black Hat
        在原始图像和Close操作图像之间取差集
"""
# 加载图像
# img = cv.imread('./datas/j.png', 0)
# # a. 定义一个核（全部设置为1表示对核中9*9区域的所有像素均进行考虑，设置为0表示不考虑）
# kernel = np.ones((9, 9), np.uint8)
#
# # b1. Close(先膨胀,再腐蚀)
# dst1 = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)
# # b2. Black Hat(close - src)
# dst2 = cv.morphologyEx(img, op=cv.MORPH_BLACKHAT, kernel=kernel, iterations=1)
# # dst2 = dst1 - img
#
#
# # c. 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(132)
# plt.imshow(dst1, 'gray')
# plt.title('Close')
#
# plt.subplot(133)
# plt.imshow(dst2, 'gray')
# plt.title('BlackHat')
# plt.show()
#
# img0 = cv.imread("./datas/car.jpg", 0)
# print(img.shape)
# img = cv.GaussianBlur(img0, (5, 5), 0)
#
# # 普通二值化操作
# ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# # 使用均值的方式产生当前像素点对应的阈值，
# # 使用(x,y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值
# th2 = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
#                            thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
# # 使用高斯分布的方式产生当前像素点对应的阈值
# # 使用(x,y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值，
# # 其中权重为和当前数据有关的高斯随机数
# th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv.THRESH_BINARY, 11, 2)
#
# ret4, th4 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# th5 = cv.adaptiveThreshold(th4, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv.THRESH_BINARY, 11, 2)
#
# ret6, th6 = cv.threshold(cv.GaussianBlur(th3, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#
# # 对th3做一些形态学变换
# kernel = np.ones((5, 5), np.uint8)
# th7 = cv.erode(th3, kernel, iterations=1)
# th7 = cv.dilate(th7, kernel, iterations=3)
# th7 = cv.erode(th7, kernel, iterations=3)
# th7 = cv.dilate(th7, kernel, iterations=3)
#
# titles = ['Original Image', 'GaussianBlur', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
#           f'OTSU:{ret4}', f'OTSU:{ret4} + Adaptive Gaussian Thresholding',
#           f'Adaptive Gaussian Thresholding + OTSU:{ret6}', 'morph']
# images = [img0, img, th1, th2, th3, th4, th5, th6, th7]
# plt.figure(figsize=(20, 10))
# for i in range(len(images)):
#     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

"""
16. 图像梯度
    通过对图像梯度的操作，可以发现图像的边缘信息
    在OpenCV中提供了三种类型的高通滤波器，常见处理方式：Sobel、Scharr以及Laplacian导数
"""
#  Sobel
#  Sobel滤波器通过加入高斯平滑以及方向导数，抗噪性比较强。
# 加载图像
# img = cv.imread('./datas/xiaoren.png', 0)
#
# # 画几条线条
# rows, cols = img.shape
# cv.line(img, pt1=(0, rows // 3), pt2=(cols, rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(0, 2 * rows // 3), pt2=(cols, 2 * rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(cols // 3, 0), pt2=(cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(2 * cols // 3, 0), pt2=(2 * cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(0, 0), pt2=(cols, rows), color=0, thickness=1)
# cv.line(img, pt1=(0, rows), pt2=(cols, 0), color=0, thickness=5)
# print("")
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # x方向的的Sobel过滤，ksize：一般取值为3,5,7；
# # 第二个参数：ddepth，给定输出的数据类型的取值范围，默认为unit8的，取值为[0,255]，如果给定-1，表示输出的数据类型和输入一致。
# sobelx = cv.Sobel(img, 6, dx=1, dy=0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, dx=0, dy=1, ksize=5)
# sobelx2 = cv.Sobel(img, cv.CV_64F, dx=2, dy=0, ksize=5)
# sobely2 = cv.Sobel(img, cv.CV_64F, dx=0, dy=2, ksize=5)
# sobel = cv.Sobel(img, cv.CV_64F, dx=1, dy=1, ksize=5)
# sobelx_y = cv.Sobel(sobelx, cv.CV_64F, dx=0, dy=1, ksize=5)
# sobely_x = cv.Sobel(sobely, cv.CV_64F, dx=1, dy=0, ksize=5)
#
# plt.figure(figsize=(20, 20))
# # c. 可视化
# plt.subplot(331)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(332)
# plt.imshow(sobelx, 'gray')
# plt.title('sobelx')
#
# plt.subplot(333)
# plt.imshow(sobely, 'gray')
# plt.title('sobely')
#
# plt.subplot(334)
# plt.imshow(sobelx2, 'gray')
# plt.title('sobelx2')
#
# plt.subplot(335)
# plt.imshow(sobely2, 'gray')
# plt.title('sobely2')
#
# plt.subplot(336)
# plt.imshow(sobel, 'gray')
# plt.title('sobel')
#
# plt.subplot(337)
# plt.imshow(sobelx_y, 'gray')
# plt.title('sobelx_y')
#
# plt.subplot(338)
# plt.imshow(sobely_x, 'gray')
# plt.title('sobely_x')
#
# plt.show()
#
# # 基于
# # 自定义一个kernel核
# kernel = np.asarray([
#     [-1, -2, -1],
#     [0, 0, 0],
#     [1, 2, 1]
# ])
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# sobely = cv.filter2D(img, 6, kernel)
# sobelx = cv.filter2D(img, 6, kernel.T)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('Original')
#
# plt.subplot(132)
# plt.imshow(sobelx, 'gray')
# plt.title('sobelx')
#
# plt.subplot(133)
# plt.imshow(sobely, 'gray')
# plt.title('sobely')
# plt.show()
#
# # 基于
# # 自定义一个kernel核
# kernel = np.asarray([
#     [-1, -2, -1],
#     [0, 0, 0],
#     [1, 2, 1]
# ])
# kernel1 = np.asarray([[1, 2, 1]])
# kernel2 = np.asarray([[-1], [0], [1]])
#
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# # sobelx = cv.filter2D(img, 6, kernel.T)
# # 先高斯平滑
# a = cv.filter2D(img, 6, kernel1.T)
# sobelx = cv.filter2D(a, 6, kernel2.T)
#
# # sobely = cv.filter2D(img, 6, kernel)
# # 先高斯平滑
# a = cv.filter2D(img, 6, kernel1)
# # 再垂直梯度
# sobely = cv.filter2D(a, 6, kernel2)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('Original')
#
# plt.subplot(132)
# plt.imshow(sobelx, 'gray')
# plt.title('sobelx')
#
# plt.subplot(133)
# plt.imshow(sobely, 'gray')
# plt.title('sobely')
# plt.show()

#  Scharr
#  Scharr可以认为是一种特殊的Sobel方式, 实际上就是一种特殊的kernel
# 加载图像
# img = cv.imread('./datas/xiaoren.png', 0)
#
# # 画几条线条
# rows, cols = img.shape
# cv.line(img, pt1=(0, rows // 3), pt2=(cols, rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(0, 2 * rows // 3), pt2=(cols, 2 * rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(cols // 3, 0), pt2=(cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(2 * cols // 3, 0), pt2=(2 * cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(0, 0), pt2=(cols, rows), color=0, thickness=1)
# cv.line(img, pt1=(0, rows), pt2=(cols, 0), color=0, thickness=5)
# print("")
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # Scharr中，dx和dy必须有一个为0，一个为1
# scharr_x = cv.Scharr(img, cv.CV_64F, dx=1, dy=0)
# scharr_y = cv.Scharr(img, cv.CV_64F, dx=0, dy=1)
# scharr_x_y = cv.Scharr(scharr_x, cv.CV_64F, dx=0, dy=1)
# scharr_y_x = cv.Scharr(scharr_y, cv.CV_64F, dx=1, dy=0)
#
# plt.figure(figsize=(20, 10))
# # c. 可视化
# plt.subplot(231)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(232)
# plt.imshow(scharr_x, 'gray')
# plt.title('scharr_x')
#
# plt.subplot(233)
# plt.imshow(scharr_y, 'gray')
# plt.title('scharr_y')
#
# plt.subplot(234)
# plt.imshow(scharr_x_y, 'gray')
# plt.title('scharr_x_y')
#
# plt.subplot(235)
# plt.imshow(scharr_y_x, 'gray')
# plt.title('scharr_y_x')
#
# plt.show()
#
# # 基于
# # 自定义一个kernel核
# kernel = np.asarray([
#     [-3, -10, -3],
#     [0, 0, 0],
#     [3, 10, 3]
# ])
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# scharr_y = cv.filter2D(img, 6, kernel)
# scharr_x = cv.filter2D(img, 6, kernel.T)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('Original')
#
# plt.subplot(132)
# plt.imshow(scharr_x, 'gray')
# plt.title('scharr_x')
#
# plt.subplot(133)
# plt.imshow(scharr_y, 'gray')
# plt.title('scharr_y')
# plt.show()

#  Laplacian
#  使用拉普拉斯算子进行边缘提取

# 加载图像
# img = cv.imread('./datas/xiaoren.png', 0)
#
# # 画几条线条
# rows, cols = img.shape
# cv.line(img, pt1=(0, rows // 3), pt2=(cols, rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(0, 2 * rows // 3), pt2=(cols, 2 * rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(cols // 3, 0), pt2=(cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(2 * cols // 3, 0), pt2=(2 * cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(0, 0), pt2=(cols, rows), color=0, thickness=1)
# cv.line(img, pt1=(0, rows), pt2=(cols, 0), color=0, thickness=5)
# print("")
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # ksize设置为3
# ksize = 3
# laplacian = cv.Laplacian(img, cv.CV_64F, ksize=ksize)
# laplacian = np.uint8(np.absolute(laplacian))
#
# sobel_x = cv.Sobel(img, cv.CV_64F, dx=1, dy=0, ksize=ksize)
# sobel_x = np.uint8(np.absolute(sobel_x))
# sobel_y = cv.Sobel(img, cv.CV_64F, dx=0, dy=1, ksize=ksize)
# sobel_y = np.uint8(np.absolute(sobel_y))
# scharr_x = cv.Scharr(img, cv.CV_64F, dx=1, dy=0)
# scharr_x = np.uint8(np.absolute(scharr_x))
# scharr_y = cv.Scharr(img, cv.CV_64F, dx=0, dy=1)
# scharr_y = np.uint8(np.absolute(scharr_y))
#
# plt.figure(figsize=(20, 10))
# # c. 可视化
# plt.subplot(231)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(232)
# plt.imshow(sobel_x, 'gray')
# plt.title('sobel_x')
#
# plt.subplot(233)
# plt.imshow(sobel_y, 'gray')
# plt.title('sobel_y')
#
# plt.subplot(234)
# plt.imshow(laplacian, 'gray')
# plt.title('laplacian')
#
# plt.subplot(235)
# plt.imshow(scharr_x, 'gray')
# plt.title('scharr_x')
#
# plt.subplot(236)
# plt.imshow(scharr_y, 'gray')
# plt.title('scharr_y')
# plt.show()
# #  在Sobel检测中，ddepth对于结果的影响，
# #  当输出的depth设置为比较低的数据格式，那么当梯度值计算为负值的时候，就会将其重置为0，从而导致失真。
# #  在Laplacian检测中，该问题不大。
# np.uint8([0, 255, 0, 255, 0])
# # 构建一个图像
# # 构建黑底白框的图像
# # img = np.zeros((300,300), np.uint8)
# # img[100:200,100:200] = 255
# # 构建白底黑框的图像
# img = np.ones((300, 300), np.uint8) * 255
# img[100:200, 100:200] = 0
#
# ksize = 5
# # 做Sobel的操作
# # np.unique(dst2)、 np.uint8(np.absolute(np.unique(dst2)))
# dst1 = cv.Laplacian(img, cv.CV_8U, ksize=ksize)
# dst2 = cv.Laplacian(img, cv.CV_64F, ksize=ksize)
# dst3 = np.uint8(np.absolute(dst2))
#
# dst4 = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=ksize)
# dst5 = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
# dst6 = np.uint8(np.absolute(dst5))
#
# plt.figure(figsize=(20, 10))
# # c. 可视化
# plt.subplot(241)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(242)
# plt.imshow(dst1, 'gray')
# plt.title('Laplacian1')
#
# plt.subplot(243)
# plt.imshow(dst2, 'gray')
# plt.title('Laplacian2')
#
# plt.subplot(244)
# plt.imshow(dst3, 'gray')
# plt.title('Laplacian3')
#
# plt.subplot(245)
# plt.imshow(dst4, 'gray')
# plt.title('Sobel1')
#
# plt.subplot(246)
# plt.imshow(dst5, 'gray')
# plt.title('Sobel2')
#
# plt.subplot(247)
# plt.imshow(dst6, 'gray')
# plt.title('Sobel3')
#
# plt.show()

"""
17. 特定卷积核的应用代码
"""
# 加载图像
# img = cv.imread('./datas/xiaoren.png', 0)
#
# # 画几条线条
# rows, cols = img.shape
# cv.line(img, pt1=(0, rows // 3), pt2=(cols, rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(0, 2 * rows // 3), pt2=(cols, 2 * rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(cols // 3, 0), pt2=(cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(2 * cols // 3, 0), pt2=(2 * cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(0, 0), pt2=(cols, rows), color=0, thickness=1)
# cv.line(img, pt1=(0, rows), pt2=(cols, 0), color=0, thickness=5)
# print("")
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 基于
# # 自定义一个kernel核
# kernel = np.asarray([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 1],
# ])
# # 做一个卷积操作
# # 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
# dst1 = cv.filter2D(img, 6, kernel)
# dst2 = dst1 + 0
#
# for i in range(100):
#     dst1 = cv.filter2D(dst1, 6, kernel)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('Original')
#
# plt.subplot(132)
# plt.imshow(dst1, 'gray')
# plt.title('dst1')
#
# plt.subplot(133)
# plt.imshow(dst2, 'gray')
# plt.title('dst2')
#
# plt.show()


"""
18. Canny
    Canny算法是一种比Sobel和Laplacian效果更好的一种边缘检测算法；在Canny算法中，主要包括以下几个阶段：
        1. Noise Reduction：降噪，使用5*5的kernel做Gaussian filter降噪；
        2. Finding Intensity Gradient of the Image：求图像像素的梯度值；
        3. Non-maximum Suppression：删除可能不构成边缘的像素，
              即在渐变方向上相邻区域的像素梯度值是否是最大值，如果不是，则进行删除。
        4. Hysteresis Thresholding：基于阈值来判断是否属于边；
              大于maxval的一定属于边，小于minval的一定不属于边，在这个中间的可能属于边的边缘。
    参考页面：
        https://en.wikipedia.org/wiki/Canny_edge_detector
        http://dasl.unlv.edu/daslDrexel/alumni/bGreen/www.pages.drexel.edu/_weg22/can_tut.html
"""
# 加载图像
# img = cv.imread('./datas/xiaoren.png', 0)
# # 画几条线条
# rows, cols = img.shape
# cv.line(img, pt1=(0, rows // 3), pt2=(cols, rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(0, 2 * rows // 3), pt2=(cols, 2 * rows // 3), color=0, thickness=5)
# cv.line(img, pt1=(cols // 3, 0), pt2=(cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(2 * cols // 3, 0), pt2=(2 * cols // 3, rows), color=0, thickness=5)
# cv.line(img, pt1=(0, 0), pt2=(cols, rows), color=0, thickness=1)
# cv.line(img, pt1=(0, rows), pt2=(cols, 0), color=0, thickness=5)
# print("")
#
# img = cv.imread('./datas/car3.jpg', 0)
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # 做一个Canny边缘检测(OpenCV中是不包含高斯去燥的)
# # a. 高斯去燥
# blur = cv.GaussianBlur(img, (5, 5), 0)
# # b. Canny边缘检测
# edges = cv.Canny(blur, threshold1=50, threshold2=250)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(132)
# plt.imshow(blur, cmap='gray')
# plt.title('Gaussian Blur Image')
#
# plt.subplot(133)
# plt.imshow(edges, cmap='gray')
# plt.title('Canny Edge Image')
# plt.show()


"""
19. 轮廓信息
    轮廓信息可以简单的理解为图像曲线的连接点信息，在目标检测以及识别中有一定的作用。
    轮廓信息的查找最好是基于灰度图像或者边缘特征图像，因为基于这样的图像比较容易找连接点信息；
    NOTE:在OpenCV中，查找轮廓是在黑色背景中查找白色图像的轮廓信息。
"""
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
#
# # 将图像转换为灰度图像
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # 做一个图像反转(0 -> 255, 255 -> 0)
# img1 = cv.bitwise_not(img1)
#
# # 做一个二值化
# ret, thresh = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
# # thresh = img1
#
# # 发现轮廓信息
# # 第一个参数是原始图像，第二个参数是轮廓的检索模型，第三个参数是轮廓的近似方法
# # 第一个返回值为轮廓，第三个返回值为层次信息
# # CHAIN_APPROX_SIMPLE指的是对于一条直线上的点而言，仅仅保留端点信息,
# # 而CHAIN_APPROX_NONE保留所有点
# # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# # 在图像中绘制图像
# # 当contourIdx为-1的时候，表示绘制所有轮廓，当为大于等于0的时候，表示仅仅绘制某一个轮廓
# # 这里的返回值img3和img是同一个对象，在当前版本中
# img3 = cv.drawContours(copy.deepcopy(img), contours, contourIdx=-1, color=(0, 0, 255), thickness=5)
# max_idx = np.argmax([len(t) for t in contours])
# img3 = cv.drawContours(img3, contours, contourIdx=max_idx, color=(0, 255, 0), thickness=2)
#
# # 将轮廓当成边缘
# img4 = np.zeros_like(img3)
# cv.drawContours(img4, contours, contourIdx=-1, color=(255, 255, 255), thickness=2)
#
# plt.figure(figsize=(20, 20))
# # 可视化
# plt.subplot(231)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(232)
# plt.imshow(thresh, cmap='gray')
# plt.title('thresh')
#
# plt.subplot(233)
# plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
# plt.title('img3')
#
# plt.subplot(234)
# plt.imshow(img1, cmap='gray')
# plt.title('img1')
#
# plt.subplot(235)
# plt.imshow(img4, cmap='gray')
# plt.title('img4')
#
# plt.show()
# print("总的轮廓数目:{}".format(len(contours)))
# print(contours[0])
# print(hierarchy.shape)
#
#
# # 加载图像
# # img = cv.imread('car3_plat.jpg')
# img = cv.imread('./datas/car.jpg')
#
# # 将图像转换为灰度图像
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # 做一个图像反转(0 -> 255, 255 -> 0)
# img1 = cv.bitwise_not(img1)
#
# # 做一个二值化
# ret, thresh = cv.threshold(img1, 200, 255, cv.THRESH_BINARY)
#
# # 发现轮廓信息
# # 第一个参数是原始图像，第二个参数是轮廓的检索模型，第三个参数是轮廓的近似方法
# # 第一个返回值为轮廓，第三个返回值为层次信息
# # CHAIN_APPROX_SIMPLE指的是对于一条直线上的点而言，仅仅保留端点信息,
# # 而CHAIN_APPROX_NONE保留所有点
# # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# # 在图像中绘制图像
# # 当contourIdx为-1的时候，表示绘制所有轮廓，当为大于等于0的时候，表示仅仅绘制某一个轮廓
# # 这里的返回值img3和img是同一个对象，在当前版本中
# img3 = cv.drawContours(copy.deepcopy(img), contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
# # max_idx = np.argmax([len(t) for t in contours])
# # img3 = cv.drawContours(img, contours, contourIdx=max_idx, color=(0, 0, 255), thickness=2)
#
# # 将轮廓当成边缘
# img4 = np.zeros_like(img3)
# cv.drawContours(img4, contours, contourIdx=-1, color=(255, 255, 255), thickness=2)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(231)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(232)
# plt.imshow(thresh, cmap='gray')
# plt.title('thresh')
#
# plt.subplot(233)
# plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
# plt.title('img3')
#
# plt.subplot(234)
# plt.imshow(img1, cmap='gray')
# plt.title('img1')
#
# plt.subplot(235)
# plt.imshow(img4, cmap='gray')
# plt.title('img4')
# plt.show()
#
#
# # 加载图像
# img = cv.imread('./datas/car2.jpg')
#
# # 将图像转换为灰度图像
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img1 = cv.bitwise_not(img1)
#
# # 做一个二值化
# ret, thresh = cv.threshold(img1, 200, 255, cv.THRESH_BINARY)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# mask = np.zeros_like(thresh)
# cv.drawContours(mask, contours, contourIdx=-1, color=(255, 255, 255), thickness=2)
# # mask = np.uint8(mask[:,:,0])
# kernel = np.ones((5, 5), np.uint8)
# mask = cv.dilate(mask, kernel, iterations=4)  # 膨胀
# # mask = cv.erode(mask, kernel, iterations=4) # 腐蚀
# img_mask = cv.bitwise_and(img, img, mask=mask)
#
# # 做第二次轮廓查询
# img1 = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)
# img1 = cv.bitwise_not(img1)
# ret, thresh = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# img3 = cv.drawContours(copy.deepcopy(img), contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(231)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(232)
# plt.imshow(mask, cmap='gray')
# plt.title('mask')
#
# plt.subplot(233)
# plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
# plt.title('img3')
# plt.show()

"""
20. 轮廓信息说明
"""
# 构建黑底白框的图像
# img = np.zeros((300, 300), np.uint8)
# img[10:290, 10:290] = 255
# img[50:200, 50:200] = 0
# img[55:100, 55:100] = 255
# img[120:190, 120:150] = 255
# img[130:160, 130:145] = 0
# img[210:250, 210:250] = 0
# img[250:270, 150:180] = 0
# img[205:220, 205:220] = 0
#
# ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# img3 = cv.drawContours(img, contours, contourIdx=-1, color=128, thickness=2)
#
# # c. 可视化
# plt.imshow(img, 'gray')
# plt.title('img')
# plt.show()
# print("总的轮廓数目:{}".format(len(contours)))
# # contours
# print(contours[0])
# print(hierarchy)
# # 是一个[1, n, 4]格式，n为轮廓的数目，这个中间保存的是轮廓包含信息
# # 每个轮廓的层次信息是一个4维的向量值，
# # 第一个值表示当前轮廓的上一个同层级的轮廓下标，
# # 第二值表示当前轮廓的下一个同层级的轮廓下标，
# # 第三个表示当前轮廓的第一个子轮廓的下标，
# # 第四个就表示当前轮廓的父轮廓的下标
# print(hierarchy.shape)

"""
21. 轮廓属性
        当获取得到轮廓坐标后，就可以基于轮廓来计算面积、周长等属性。
"""
# 加载图像
# img = cv.imread('./datas/xiaoren.png')
#
# # 将图像转换为灰度图像
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # 做一个图像反转(0 -> 255, 255 -> 0)
# img1 = cv.bitwise_not(img1)
#
# # 做一个二值化
# ret, thresh = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
# # 发现轮廓信息
# # 第一个参数是原始图像，
# # 第二个参数是轮廓的检索模型(RETR_LIST表示检索所有轮廓，但是不保留层次信息)，第三个参数是轮廓的近似方法
# # 第一个返回值是修改过的图像(一般就是原始图像)，第二个参数值为轮廓，第三个参数值为层次信息
# # CHAIN_APPROX_SIMPLE指的是对于一条直线上的点而言，仅仅保留端点信息,
# # 而CHAIN_APPROX_NONE保留所有点
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# idx = np.argmax([len(t) for t in contours])
# cnt = contours[idx]  # 最大区域的轮廓
#
# # 绘制轮廓
# img3 = cv.drawContours(img, contours, contourIdx=idx, color=(0, 0, 255), thickness=2)
# print(cnt)
#
# # 计算面积
# area = cv.contourArea(cnt)
# # 计算周长
# perimeter = cv.arcLength(cnt, closed=True)
# print("面积为:{}, 周长为:{}".format(area, perimeter))
# # 获取最大的矩形边缘框, 返回值为矩形框的左上角的坐标以及宽度和高度
# x, y, w, h = cv.boundingRect(cnt)
# # 绘图
# cv.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# print(x, y, w, h)
# # 绘制最小矩形(所有边缘在矩形内)、得到矩形的点(左下角、左上角、右上角、右下角<顺序不一定>)、绘图
# # minAreaRect：求得一个包含点集cnt的最小面积的矩形，这个矩形可以有一点的旋转偏转的，输出为矩形的四个坐标点
# # rect为三元组，第一个元素为旋转中心点的坐标
# # rect为三元组，第二个元素为矩形的高度和宽度
# # rect为三元组，第三个元素为旋转大小，正数表示顺时针选择，负数表示逆时针旋转
# rect = cv.minAreaRect(cnt)
# box = cv.boxPoints(rect)
# box = np.int64(box)
# cv.drawContours(img, [box], 0, (0, 255, 0), 2)
# print(rect)
# print(box)
# # 绘制最小的圆(所有边缘在圆内)
# (x, y), radius = cv.minEnclosingCircle(cnt)
# center = (int(x), int(y))
# radius = int(radius)
# cv.circle(img, center, radius, (0, 0, 255), 5)
# print()
# # 绘制最小的椭圆(所有边缘不一定均在圆内)
# ellipse = cv.fitEllipse(cnt)
# cv.ellipse(img, ellipse, (0, 255, 0), 5)
# print()
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(132)
# plt.imshow(thresh, cmap='gray')
# plt.title('thresh')
#
# plt.subplot(133)
# plt.imshow(thresh, cmap='gray')
# plt.title('thresh')
# plt.show()
#
# # 旋转后提取最小矩形
# # 加载图像
# img = cv.imread('./datas/xiaoren.png')
# rows, cols, _ = img.shape
# # 旋转
# rect = [(302.3465576171875, 419.5862731933594), (372.9076232910156, 317.0608215332031), 70.12313079833984]
# M = cv.getRotationMatrix2D(center=rect[0], angle=rect[-1], scale=1)
# img = cv.warpAffine(img, M, (cols, rows), borderValue=[255, 255, 255])
#
# # 将图像转换为灰度图像
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # 做一个图像反转(0 -> 255, 255 -> 0)
# img1 = cv.bitwise_not(img1)
#
# # 做一个二值化
# ret, thresh = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
# # 发现轮廓信息
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# idx = np.argmax([len(t) for t in contours])
# cnt = contours[idx]
#
# # 绘制轮廓
# img3 = cv.drawContours(img, contours, contourIdx=idx, color=(0, 0, 255), thickness=2)
#
# # 绘制最小矩形(所有边缘在矩形内)、得到矩形的点(左下角、左上角、右上角、右下角<顺序不一定>)、绘图
# # minAreaRect：求得一个包含点集cnt的最小面积的矩形，这个矩形可以有一点的旋转偏转的，输出为矩形的四个坐标点
# # rect为三元组，第一个元素为旋转中心点的坐标
# # rect为三元组，第二个元素为矩形的高度和宽度
# # rect为三元组，第三个元素为旋转大小，正数表示顺时针选择，负数表示逆时针旋转
# rect = cv.minAreaRect(cnt)
# box = cv.boxPoints(rect)
# box = np.int64(box)
# cv.drawContours(img, [box], 0, (0, 255, 0), 2)
# print(rect)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(132)
# plt.imshow(thresh, cmap='gray')
# plt.title('thresh')
#
# plt.subplot(133)
# plt.imshow(thresh, cmap='gray')
# plt.title('thresh')
# plt.show()


"""
22. 直方图
    OpenCV中的直方图的主要功能是可以查看图像的像素信息
    以及提取直方图中各个区间的像素值的数目作为当前图像的特征属性进行机器学习模型
"""
# 加载图像
# img = cv.imread('./datas/koala.png', 0)
#
# # 两种方式基本结果基本类似
# # 基于OpenCV的API计算直方图
# hist1 = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
# # 基于NumPy计算直方图
# hist2, bins = np.histogram(img.ravel(), 256, [0, 256])
# # 和np.histogram一样的计算方式，但是效率快10倍
# hist3 = np.bincount(img.ravel(), minlength=256)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('Original Image')
#
# plt.subplot(132)
# plt.plot(hist1)
# plt.title('Hist')
#
# plt.subplot(133)
# # 可以直接使用matpliab中的hist API直接画直方图
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title('Hist2')
# plt.show()
#
#
# print(hist3)
# # 针对彩色图像计算直方图
# img = cv.imread('./datas/koala.png')
#
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()
#
# # 针对HSV图像计算直方图
# img = cv.imread('./datas/koala.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#
# # h -> 红色; s -> 蓝色; v -> 绿色
# color = ('r', 'b', 'g')
# for i, col in enumerate(color):
#     histr = cv.calcHist([img], [i], None, [256], [0, 256])
#     print(np.shape(histr))
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()
#
# # 针对HSV图像计算直方图
# img = cv.imread('./datas/tt1.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#
# chistr = cv.calcHist([img], [0], None, [180], [0, 180])
# print(np.shape(histr))
#
# plt.figure(figsize=(20, 10))
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img, cv.COLOR_HSV2RGB))
# plt.title('img')
#
# plt.subplot(122)
# plt.plot(chistr, color='r')
# plt.title('Hist')
# plt.show()
#
#
# # 加入mask位置信息的hist
# img = cv.imread('./datas/koala.png', 0)
#
# # 创建一个mask
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[50:250, 50:450] = 255
#
# # 构建mask区域的图像
# masked_img = cv.bitwise_and(img, img, mask=mask)
#
# # 计算直方图
# hist1 = cv.calcHist([img], [0], None, [256], [0, 256])
# hist2 = cv.calcHist([masked_img], [0], mask, [256], [0, 256])
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(221)
# plt.imshow(img, 'gray')
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(masked_img, 'gray')
# plt.title('masked_img')
#
# plt.subplot(223)
# plt.imshow(mask, 'gray')
# plt.title('mask')
#
# plt.subplot(224)
# plt.plot(hist1, color='r')
# plt.plot(hist2, color='g')
# plt.title('Hist')
# plt.show()
#
# # 加入mask位置信息的hist
# img = cv.imread('./datas/t.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#
# # 创建一个mask
# mask = np.zeros(img.shape[:2], np.uint8)
# # mask[95:300, 120:340] = 255 # 第一个灯
# mask[320:530, 120:340] = 255  # 第二个灯
# # mask[550:770, 120:340] = 255 # 第三个灯
#
# # 构建mask区域的图像
# masked_img = cv.bitwise_and(img, img, mask=mask)
#
# # 计算直方图
# hist1 = cv.calcHist([img], [0], None, [256], [0, 256])
# hist1 = hist1 / np.sum(hist1)
# hist2 = cv.calcHist([masked_img], [0], mask, [256], [0, 256])
# hist2 = hist2 / np.sum(hist2)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(221)
# plt.imshow(cv.cvtColor(img, cv.COLOR_HSV2RGB))
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(cv.cvtColor(masked_img, cv.COLOR_HSV2RGB))
# plt.title('masked_img')
#
# plt.subplot(223)
# plt.imshow(mask, 'gray')
# plt.title('mask')
#
# plt.subplot(224)
# plt.plot(hist1, color='r')
# plt.plot(hist2, color='g')
# plt.title('Hist')
# plt.show()


"""
23. 直方图均衡化
        如果某一个图像的像素仅限制于某个特定的范围，但是实际上一个比较好的图像像素点应该是在某一个范围内，
        所以需要将像素的直方图做一个拉伸的操作；
        比如：一个偏暗的图像，像素基本上都在较小的位置，如果将像素值增大，不就可以让图像变亮嘛！
        实际上，直方图均衡化操作是一个提高图像对比度的方式
"""
# img = cv.imread('./datas/tsukuba.png', 0)
# # img = cv.GaussianBlur(img, (5,5),0)
#
# # 做一个直方图均衡（针对一个图像，如果像素点集中在某个区域，通过该方式可以让图像变的更好）
# img2 = cv.equalizeHist(img)
#
# # 做一个自适应的直方图均衡
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img3 = clahe.apply(img)
#
# plt.figure(figsize=(20, 10))
# # 可视化
# plt.subplot(221)
# plt.imshow(img, 'gray')
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(img2, 'gray')
# plt.title('img2')
#
# plt.subplot(223)
# plt.imshow(img3, 'gray')
# plt.title('img3')
#
# plt.subplot(224)
# plt.plot(cv.calcHist([img], [0], None, [256], [0, 256]), color='r', label='image')
# # plt.plot(cv.calcHist([img2], [0], None, [256], [0,256]), color='g')
# plt.plot(cv.calcHist([img3], [0], None, [256], [0, 256]), color='b', label='img3')
# plt.legend(loc='upper left')
# plt.title('Hist')
# plt.show()

"""
傅里叶变换
也就是频域的变换, 不过现在一般不做频域变换了，可以省略不掌握。 参考网站：
http://cns-alumni.bu.edu/~slehar/fourier/fourier.html
http://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
https://dsp.stackexchange.com/questions/1637/what-does-frequency-domain-denote-in-case-of-images
https://docs.opencv.org/3.4.0/de/dbc/tutorial_py_fourier_transform.html
"""

"""
24. 模板匹配
    就是基于模板在原始图像中查找最匹配的位置
"""
plt.figure(figsize=(16, 16))
# 加载图像
img = cv.imread('./datas/xiaoren.png', 0)
# 图像copy
img2 = img.copy()
# 加载模板图像
template = cv.imread('./datas/template.png', 0)
# 得到模板图像的高度和宽度
h, w = template.shape

# 6中匹配方式
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

img2 = img2[:, :400]
# 遍历
for idx, meth in enumerate(methods):
    # 图像copy
    img = img2.copy()

    # 得到对应的方式(eval的意思是执行)
    method = eval(meth)

    # 使用给定的方式进行模板匹配（返回值为各个局部区域和模板template之间的相似度<可以认为是相似度>）
    res = cv.matchTemplate(img, template, method)

    # 从数据中查找全局最小值和最大值, 以及对应的位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # 如果求解方式为TM_SQDIFF和TM_SQDIFF_NORMED，
    # 那么矩形左上角的点就是最小值的位置；
    # 否则是最大值的位置
    if method in [cv.TM_CCORR, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # 基于左上角的坐标计算右下角的坐标
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 基于坐标画矩形
    print(meth, (top_left, bottom_right), res[top_left[1], top_left[0]])
    cv.rectangle(img, top_left, bottom_right, 180, 5)

    # 画图
    plt.subplot(len(methods) // 2, 4, 2 * idx + 1)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result {}'.format(meth))
    plt.subplot(len(methods) // 2, 4, 2 * idx + 2)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point {}'.format(meth))
    plt.suptitle(meth)
plt.show()
