import cv2
import time
import numpy
AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 3 # 自动拍照间隔
#H,W,3
cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 400, 0)
# left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(0)# 根据情况设定相机id
right_camera.set(3, 2560)  # width=1920
right_camera.set(4, 720)  # height=1080
counter = 1#图片命名的起始数字
utc = time.time()
pattern = (9, 6) # 棋盘格尺寸
cell_size= 3.68
dimension='cm'
folder = "./snapshot/" # 拍照文件目录
import os
if not os.path.exists(folder):
    os.mkdir(folder)
stage=0 #0表示相机标定，1表示生成深度图并支持操作
number=24 # 取样图片的数目
def shot(pos, frame):
    global counter
    path = folder + pos + "_" + str(counter) + ".jpg"

    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)
if stage==0:
    while True:
        # ret, left_frame = left_camera.read()
        ret, right_frame = right_camera.read()
        # right_frame=right_frame[:,:,[2,1,0]]
        print(right_frame.shape)
        left_frame,right_frame=numpy.split(right_frame,2,1)
        print(left_frame.shape)
        cv2.imshow("left", left_frame)
        cv2.imshow("right", right_frame)

        now = time.time()
        if AUTO and now - utc >= INTERVAL:
            shot("left", left_frame)
            shot("right", right_frame)
            counter += 1
            utc = now

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot("left", left_frame)
            shot("right", right_frame)
            counter += 1

    # left_camera.release()
    right_camera.release()
    cv2.destroyWindow("left")
    cv2.destroyWindow("right")
if stage==1:
   left_pics=[]
   right_pics=[]
   for i in range(number):
       pic1= cv2.imread(folder + "left" + "_" + str(counter) + ".jpg")
       pic2 = cv2.imread(folder + "right" + "_" + str(counter) + ".jpg")
       left_pics.append(pic1)
       right_pics.append(pic2)
       counter+=1
   tmp=[]
   coins=[]
   a1=[]
   for i in range(number):
    ii=i
    gray=cv2.cvtColor(left_pics[i], cv2.COLOR_BGR2GRAY)
    ok, corners=cv2.findChessboardCorners(cv2.cvtColor(left_pics[i], cv2.COLOR_BGR2GRAY),pattern,None)
    tmps=[]
    for ih in range(pattern[1]):
        for j in range(pattern[0]):
            tmps.append([j,ih,0])
    tmps=numpy.array(tmps,dtype=numpy.float32)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners=cv2.cornerSubPix(cv2.cvtColor(left_pics[i], cv2.COLOR_BGR2GRAY),corners,pattern,(-1,-1),criteria)
        a1.append(ii)
        tmp.append(corners)
        coins.append(tmps)
   tmp1=tmp
   tmp=[]
   coins=[]
   a2=[]
   for i in range(number):
    ii=i
    gray=cv2.cvtColor(right_pics[i], cv2.COLOR_BGR2GRAY)
    ok, corners=cv2.findChessboardCorners(cv2.cvtColor(right_pics[i], cv2.COLOR_BGR2GRAY),pattern,None)
    tmps=[]
    for ih in range(pattern[1]):
        for j in range(pattern[0]):
            tmps.append([j,ih,0])
    tmps=numpy.array(tmps,dtype=numpy.float32)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners=cv2.cornerSubPix(cv2.cvtColor(right_pics[i], cv2.COLOR_BGR2GRAY),corners,pattern,(-1,-1),criteria)
        print(ii)
        a2.append(ii)
        tmp.append(corners)
        coins.append(tmps)
   # print(tmp)
   u1=[]
   u2=[]
   u3=[]
   for item in a2:
       if item in a1:
           print(item)
           u1.append(tmp[a2.index(item)])
           u2.append(coins[a2.index(item)])
           u3.append(tmp1[a1.index(item)])
   tmp1=u3
   coins=u2
   tmp=u1
   ret, mtx1, dist1, rvecs, tvecs = cv2.calibrateCamera(coins, tmp1, gray.shape[::-1], None, None)
   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(coins, tmp, gray.shape[::-1], None, None)
   print(ret)
   print(mtx)
   print(dist)
   print(rvecs)
   print(tvecs)
   retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(coins,tmp1,tmp,mtx1,dist1,mtx,dist,gray.shape[::-1])
   print(R)
   print(T)
   print(cameraMatrix1)
   print(cameraMatrix2)
   R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                                     cameraMatrix2, distCoeffs2, gray.shape[::-1], R,
                                                                     T)
   # 计算更正map
   left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray.shape[::-1], cv2.CV_16SC2)
   right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray.shape[::-1], cv2.CV_16SC2)
   class tt:
       def __init__(self):
           pass
   camera_configs = tt()
   camera_configs.left_map1=left_map1
   camera_configs.left_map2=left_map2
   camera_configs.right_map1=right_map1
   camera_configs.right_map2=right_map2
   camera_configs.Q=Q
   if True:
    import numpy as np
    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("depth")
    cv2.moveWindow("left", 0, 0)
    cv2.moveWindow("right", 600, 0)
    cv2.createTrackbar("num", "depth", 0, 20, lambda x: None)
    cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
    # camera1 = cv2.VideoCapture(0)
    camera2 = cv2.VideoCapture(0)

    camera2.set(3, 2560)  # width=1920
    camera2.set(4, 720)  # height=1080
    # 添加点击事件，打印当前点的距离
    def callbackFunc(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print(str(threeD[y][x][-1]*cell_size)+"cm")


    cv2.setMouseCallback("left", callbackFunc, None)

    while True:
        ret2, frame2 = camera2.read()
        ret1=ret2
        frame1,frame2=numpy.split(frame2,2,1)

        if not ret1 or not ret2:
            break

        # 根据更正map对图片进行重构
        img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

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
        if num<5:
            num=5
        import copy
        # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
        # stereo = cv2.StereoSGBM_create(numDisparities=16 * num, blockSize=blockSize,P1=8 *81,\
        #                                P2=32 *81,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100\
        #                                ,speckleRange=32,preFilterCap=63)
        stereo = cv2.StereoSGBM_create(numDisparities=16 * num, blockSize=blockSize)
        disparity = stereo.compute(imgL, imgR)
        # disp1=disparity
        disp = cv2.normalize(disparity, copy.deepcopy(disparity), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)#注意threeD是以左图为基准的
        # print(threeD.shape)
        cv2.imshow("left", img1_rectified)
        cv2.imshow("right", img2_rectified)
        cv2.imshow("depth", disp)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("./snapshot/BM_left.jpg", imgL)
            cv2.imwrite("./snapshot/BM_right.jpg", imgR)
            cv2.imwrite("./snapshot/BM_depth.jpg", disp)

    # camera1.release()
    camera2.release()
    cv2.destroyAllWindows()
