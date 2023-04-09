import cv2
import time
import numpy as np
import math

left_camera_matrix_tmp = np.array([[748.537220645527, 0., 0.],
                               [-0.171086862715876, 748.573756045103, 0.],
                               [620.640398806220, 347.086603859931, 1.]])
left_camera_matrix = np.transpose(left_camera_matrix_tmp)

# left_distortion = np.array([[0.0707605491596689, 0.113247907007358, -0.00257192165876870, -0.00297103442720558, -0.415081075926139]])
# left_distortion = np.array([[0.156608615700261, -0.269277128984164, 0.00103312627053495, -0.00104941393517511, 0.101856408758518]])
left_distortion = np.array([[0.120541339996878, -0.168740153755222, 0.000629843118552386, 0.00179348606459091, 0.]])
right_camera_matrix_tmp = np.array([[749.431315379533, 0., 0.],
                                [0.281799638767653, 749.479276057008, 0.],
                                [636.070081876925, 332.758745657899, 1.]])
right_camera_matrix = np.transpose(right_camera_matrix_tmp)
# right_distortion = np.array([[0.106232229280971, -0.0853015042338415, -0.00365385125267755, -0.00537841211975069, -0.128909796065427]])
# right_distortion = np.array([[0.158082004297845, -0.267853407652530, 0.000149695995072505, -0.000911479619948412, 0.0983556013356118]])
right_distortion = np.array([[0.0974510265399204, -0.0838075072715433, -0.000628940465316651, 0.000254655741183247, 0.]])
R = np.array([[0.999991101510547, -0.000820710991406780, 0.00413803494327793],
              [0.000856689010869184, 0.999961785513621, -0.00870020668427976],
              [-0.00413073645514466, 0.00870367427464484, 0.999953590458307]])  # 旋转向量

T = np.array([-59.4156595079536, -0.118520398706135, -1.39762490025346])  # 平移关系向量TranslationOfCamera2

size = (1280, 720)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
cv2.createTrackbar("numDisparities", "depth", 0, 50, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 3, 255, lambda x: None)
cv2.createTrackbar("minDisparity", "depth", 0, 10, lambda x: None)


cap = cv2.VideoCapture("test1.avi")
# 获取视频帧的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频流的总帧数
fcount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 跳到某一感兴趣帧并从此帧开始读取,如从第360帧开始读取
cap.set(cv2.CAP_PROP_POS_FRAMES, 4/5*fcount)


# camera2 = cv2.VideoCapture(1)

# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])


cv2.setMouseCallback("depth", callbackFunc, None)
blockSize = 3
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('can\'t open camera or camera has been opened')
        break
    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]
    # 将图片置为灰度图
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    # blockSize = 3

    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    minDisparity = cv2.getTrackbarPos("minDisparity", "depth")

    if blockSize % 2 == 0:
        blockSize += 1

    blockSize = 3
    # num = 13
    num = 4
    img_channels = 3
    minDisparity = 0

    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                   numDisparities=num * 16,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)
    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    cv2.imshow("left", img1_rectified)
    cv2.imshow("right", img2_rectified)
    # cv2.imshow("depth", disp)
    cv2.imshow("depth", dis_color)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # elif key == ord("s"):
    #     cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_left.jpg", imgL)
    #     cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_right.jpg", imgR)
    #     cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_depth.jpg", disp)

cap.release()
# camera2.release()
cv2.destroyAllWindows()
