# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
定位
其实是一种辅助功能
"""

"""
基于Haar Cascades的Face Detection
"""
# 加载定义好的人脸以及眼睛信息匹配信息
face_cascade = cv.CascadeClassifier('./datas/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./datas/haarcascade_eye.xml')
# 加载图像
img = cv.imread('./datas/faces.png')
# img = cv.imread('./datas/v1_frame/img_400.png')
# 转换为灰度图像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 检测图像
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    # 画人脸区域
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 获得人脸区域
    roi_gray = gray[y:y + h, x:x + w]
    roi_img = img[y:y + h, x:x + w]

    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # 画眼睛
        cv.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

# 可视化
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# 从摄像机获取视频 + 人脸区域提取
# 创建一个基于摄像头的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture(0)

# # 设置摄像头相关参数（但是实际参数会进行稍微的偏移）
# success=capture.set(cv.CAP_PROP_FRAME_WIDTH, 880)
# if success:
#     print("设置宽度成功")
# success=capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
# if success:
#     print("设置高度成功")

# # 打印属性
# size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
#         int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
# print(size)

# 遍历获取视频中的图像
# 读取当前时刻的摄像头捕获的图像, 返回为值：True/False, Image/None
success, frame = capture.read()
# 遍历以及等待任意键盘输入
while success and cv.waitKey(1) == -1:
    img = frame

    # NOTE: 特定，因为刘老师这个地方图像是一个反的图像，所以做一个旋转操作
    img = cv.rotate(img, rotateCode=cv.ROTATE_180)

    # 转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 做一个人脸检测
    # 检测图像
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"faces: {faces}")
    for (x, y, w, h) in faces:
        # 画人脸区域
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 获得人脸区域
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(f"eyes: {eyes}")
        for (ex, ey, ew, eh) in eyes:
            # 画眼睛
            cv.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    cv.imshow('frame', img)

    # 读取下一帧的图像
    success, frame = capture.read()

# 释放资源
capture.release()
cv.destroyAllWindows()

# 从摄像机获取视频 + 人脸区域提取
# 创建一个基于摄像头的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture(0)
# 遍历获取视频中的图像
# 读取当前时刻的摄像头捕获的图像, 返回为值：True/False, Image/None
success, frame = capture.read()
# 遍历以及等待任意键盘输入
while success and cv.waitKey(1) == -1:
    img = frame

    # 做一个人脸检测
    # 转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 检测图像
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        # NOTE: 检测出来人脸，那么直接将人脸发送给服务器进行业务逻辑 + 模型预测等相关处理
        pass

    cv.imshow('frame', img)

    # 读取下一帧的图像
    success, frame = capture.read()

# 释放资源
capture.release()
cv.destroyAllWindows()
