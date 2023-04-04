import cv2
import time
import numpy as np
import math
# import pcl
# import pcl.pcl_visualization
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):  # 判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2

left_camera_matrix = np.array([[725.221505461947, 0., 0.],
                               [-0.0848937730328429, 726.444401474590, 0.],
                               [616.817617761929, 353.889661421403, 1.]])

# left_distortion = np.array([[0.0707605491596689, 0.113247907007358, -0.00257192165876870, -0.00297103442720558, -0.415081075926139]])
# left_distortion = np.array([[0.156608615700261, -0.269277128984164, 0.00103312627053495, -0.00104941393517511, 0.101856408758518]])
left_distortion = np.array([[0., -0., 0.00103312627053495, -0.00104941393517511, 0.]])
right_camera_matrix = np.array([[725.864821487200, 0., 0.],
                                [1.11035482054761, 727.428955547968, 0.],
                                [633.586698574547, 339.799672365552, 1.]])

# right_distortion = np.array([[0.106232229280971, -0.0853015042338415, -0.00365385125267755, -0.00537841211975069, -0.128909796065427]])
# right_distortion = np.array([[0.158082004297845, -0.267853407652530, 0.000149695995072505, -0.000911479619948412, 0.0983556013356118]])
right_distortion = np.array([[0., -0., 0.000149695995072505, -0.000911479619948412, 0.]])
R = np.array([[0.999974988968334, -0.000727834478437193, 0.00703503338676566],
              [0.000783601209083440, 0.999968269854069, -0.00792749987095869],
              [-0.00702904025639578, 0.00793281425667620, 0.999943829947984]])  # 旋转向量

T = np.array([-60.3017099163414, 0.340461438339016, -0.377981234460957])  # 平移关系向量TranslationOfCamera2

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
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 1920,1280
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 1080,720


# camera2 = cv2.VideoCapture(1)

# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])


cv2.setMouseCallback("depth", callbackFunc, None)
blockSize = 3
while True:
    ret, frame = camera.read()
    if not ret:
        print('can\'t open camera or camera has been opened')
        break
    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]
    # 将图片置为灰度图
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    # img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_AREA)
    # img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_AREA)
    #
    # # 消除畸变
    # frame1 = cv2.undistort(frame1, left_camera_matrix, left_distortion)
    # frame2 = cv2.undistort(frame2, right_camera_matrix, right_distortion)
    # # 立体匹配
    # iml_, imr_ = preprocess(frame1, frame2)  # 预处理，一般可以削弱光照不均的影响，不做也可以

    # 转换为opencv的BGR格式
    # imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    # imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

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

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=4*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=1,
        P1=216,
        P2=864,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000.)
    wls_filter.setSigmaColor(1.2)
    displ = left_matcher.compute(frame1, frame2)  # .astype(np.float32)/16
    dispr = right_matcher.compute(frame2, frame1)  # .astype(np.float32)/16

    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, frame1, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(filteredImg, Q, handleMissingValues=True)
    # # 计算出的threeD，需要乘以16，才等于现实中的距离
    # threeD = threeD * 16

    # cv2.imshow("left", img1_rectified)
    # cv2.imshow("right", img2_rectified)
    # cv2.imshow("depth", disp)
    cv2.imshow("depth", filteredImg)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # elif key == ord("s"):
    #     cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_left.jpg", imgL)
    #     cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_right.jpg", imgR)
    #     cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_depth.jpg", disp)

camera.release()
# camera2.release()
cv2.destroyAllWindows()
