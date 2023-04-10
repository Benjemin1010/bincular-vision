# bincular-vision
双目视觉3D重建
bincularTest.py 为双目相机测试程序。

BM.py 为基于BM算法实现三维重建的算法，精度不高，但是速度较快，双目标定数据已使用matlab工具箱标定获得。

distance.py为使用opencv库实现双目标定+基于BM算法实现三维重建的算法（为某git仓库的参考算法）。

SGBM.py为基于SGBM算法实现三维重建的算法,效果尚可，但速度很慢，双目标定数据已使用matlab工具箱标定获得。
SGBM_VIDEO.py为基于SGBM.py更改的视频测距。

SGBM_WLS.py为基于SGBM算法+WLS滤波算法实现三维重建的算法,效果尚可，速度较快，可以基于此文件做进一步优化。双目标定数据已使用matlab工具箱标定获得。
SGBM_WLS_VIDEO.py为基于SGBM_WLS.py更改的视频测距。

UKF.py为基于无迹卡尔曼滤波器的单点测距算法，尚在开发中。

caliParas.mat 为使用matlab工具箱获得的双目相机的标定参数。

pycharmEvir.png为所使用的工具包版本信息。

