

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

# 双目相机的标定参数
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

# 单目图像尺寸
size = (1280, 720)

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

# 以下window为调试时使用
cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
cv2.createTrackbar("numDisparities", "depth", 0, 50, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 3, 255, lambda x: None)
cv2.createTrackbar("minDisparity", "depth", 0, 10, lambda x: None)

# 添加点击事件，打印当前点的3D坐标
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])

cv2.setMouseCallback("depth", callbackFunc, None)

cap = cv2.VideoCapture("test1.avi")
# 获取视频帧的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频流的总帧数
fcount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 跳到某一感兴趣帧并从此帧开始读取,如从第360帧开始读取
# cap.set(cv2.CAP_PROP_POS_FRAMES, 4/5*fcount)
blockSize = 3
# num = 13
num = 8
img_channels = 3
minDisparity = 0
# ------------------------------------SGBM算法----------------------------------------------------------
#   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
#   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
#   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
#                               取16、32、48、64等
#   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
#                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
# ------------------------------------------------------------------------------------------------------
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num * 16,
    blockSize=3,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=200,
    speckleRange=1,
    P1=8 * img_channels * blockSize ** 2,
    P2=32 * img_channels * blockSize ** 2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000.)
wls_filter.setSigmaColor(2.0)
wls_filter.setLRCthresh(24)
wls_filter.setDepthDiscontinuityRadius(3)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('can\'t open camera or camera has been opened')
        break

    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]
    # 将图片置为灰度图
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 三个trackbar用来调节不同的参数查看效果，确定参数后就不需要了
    # num = cv2.getTrackbarPos("num", "depth")
    # blockSize = cv2.getTrackbarPos("blockSize", "depth")
    # minDisparity = cv2.getTrackbarPos("minDisparity", "depth")
    #
    # if blockSize % 2 == 0:
    #     blockSize += 1

    # blockSize = 3
    # # num = 13
    # num = 7
    # img_channels = 3
    # minDisparity = 0
    # # ------------------------------------SGBM算法----------------------------------------------------------
    # #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    # #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    # #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    # #                               取16、32、48、64等
    # #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    # #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # # ------------------------------------------------------------------------------------------------------
    # left_matcher = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=num*16,
    #     blockSize=3,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=15,
    #     speckleWindowSize=100,
    #     speckleRange=1,
    #     P1=8 * img_channels * blockSize ** 2,
    #     P2=32 * img_channels * blockSize ** 2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    #
    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls_filter.setLambda(8000.)
    # wls_filter.setSigmaColor(2.0)
    # wls_filter.setLRCthresh(24)
    # wls_filter.setDepthDiscontinuityRadius(3)

    displ = left_matcher.compute(frame1, frame2)  # .astype(np.float32)/16
    dispr = right_matcher.compute(frame2, frame1)  # .astype(np.float32)/16

    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, frame1, None, dispr)

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(filteredImg, Q, handleMissingValues=True)

    cv2.imshow("left", frame1)
    cv2.imshow("right", frame2)
    # cv2.imshow("depth", disp)
    cv2.imshow("depth", filteredImg)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # elif key == ord("s"):
    #     cv2.imwrite("", imgL)
    #     cv2.imwrite("", imgR)
    #     cv2.imwrite("", disp)

cap.release()
cv2.destroyAllWindows()
