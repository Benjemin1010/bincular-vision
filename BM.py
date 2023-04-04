import cv2
import time
import numpy as np
left_camera_matrix = np.array([[725.221505461947,	0.,	0.],
                               [-0.0848937730328429,	726.444401474590,	0.],
                               [616.817617761929,	353.889661421403,	1.]])


# left_distortion = np.array([[0.0707605491596689, 0.113247907007358, -0.00257192165876870, -0.00297103442720558, -0.415081075926139]])
left_distortion = np.array([[0.156608615700261, -0.269277128984164, 0.00103312627053495, -0.00104941393517511, 0.101856408758518]])
right_camera_matrix = np.array([[725.864821487200,	0.,	0.],
                                [1.11035482054761,	727.428955547968,	0.],
                                [633.586698574547,	339.799672365552,	1.]])

# right_distortion = np.array([[0.106232229280971, -0.0853015042338415, -0.00365385125267755, -0.00537841211975069, -0.128909796065427]])
right_distortion = np.array([[0.158082004297845, -0.267853407652530, 0.000149695995072505, -0.000911479619948412, 0.0983556013356118]])
R = np.array([[0.999974988968334, -0.000727834478437193, 0.00703503338676566],
              [0.000783601209083440,	0.999968269854069, -0.00792749987095869],
              [-0.00702904025639578, 0.00793281425667620, 0.999943829947984]])#旋转向量

T = np.array([-60.3017099163414, 0.340461438339016 , -0.377981234460957]) # 平移关系向量TranslationOfCamera2
size = (1280, 720)# 图像尺寸

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
cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)#1920,1280
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)#1080,720
# camera2 = cv2.VideoCapture(1)

# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print (threeD[y][x])

cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    ret, frame= camera.read()
    # ret2, frame2 = camera2.read()

    if not ret :
        print('can\'t open camera or camera has been opened' )
        break
    frame1=frame[0:720 , 0:1280]
    frame2=frame[0:720 , 1280:2560]
    # frame1, frame2 = np.split(frame, 2, 1)
    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)

    cv2.imshow("left", img1_rectified)
    cv2.imshow("right", img2_rectified)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_left.jpg", imgL)
        cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_right.jpg", imgR)
        cv2.imwrite("E:/pythonlearning/opencv/screenshot/BM_depth.jpg", disp)

camera.release()
# camera2.release()
cv2.destroyAllWindows()