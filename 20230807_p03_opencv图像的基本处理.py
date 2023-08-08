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
img0 = cv.imread("./datas/car.jpg", 0)
img = cv.GaussianBlur(img0, (5, 5), 0)

# 普通二值化操作
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# 使用均值的方式产生当前像素点对应的阈值，
# 使用(x,y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值
th2 = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                           thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
# 使用高斯分布的方式产生当前像素点对应的阈值
# 使用(x,y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值，
# 其中权重为和当前数据有关的高斯随机数
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)

ret4, th4 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
th5 = cv.adaptiveThreshold(th4, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)

ret6, th6 = cv.threshold(cv.GaussianBlur(th3, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

titles = ['Original Image', 'GaussianBlur', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
          f'OTSU:{ret4}', f'OTSU:{ret4} + Adaptive Gaussian Thresholding',
          f'Adaptive Gaussian Thresholding + OTSU:{ret6}']
images = [img0, img, th1, th2, th3, th4, th5, th6]
for i in range(len(images)):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
