import math

import numpy as np
from scipy.linalg import sqrtm

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
right_distortion = np.array(
    [[0.0974510265399204, -0.0838075072715433, -0.000628940465316651, 0.000254655741183247, 0.]])
R = np.array([[0.999991101510547, -0.000820710991406780, 0.00413803494327793],
              [0.000856689010869184, 0.999961785513621, -0.00870020668427976],
              [-0.00413073645514466, 0.00870367427464484, 0.999953590458307]])  # 旋转向量

T = np.array([-59.4156595079536, -0.118520398706135, -1.39762490025346])  # 平移关系向量TranslationOfCamera2


class ukf:
    def __init__(self, f, h):
        self.f = f
        self.h = h
        self.Q = None
        self.R = None
        self.P = None
        self.x = None
        self.Z = None
        self.n = None
        self.m = None

    def GetParameter(self, Q, R, P, x0):
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0
        self.n = x0.shape[0]
        self.m = None

    def sigmas(self, x0, c):
        A = c * np.linalg.cholesky(self.P).T
        Y = (self.x * np.ones((self.n, self.n))).T
        Xset = np.concatenate((x0.reshape((-1, 1)), Y + A, Y - A), axis=1)
        return Xset

    # def world2pixel(self, intrinMat, R, T, worldPoints):
    #     a = np.zeros(3).reshape(3, 1)
    #     intrinMatTransF = np.c_[intrinMat, a]
    #     # b = np.zeros(3).reshape(1, 3)
    #     # exMat1 = np.r_[R, [b]]
    #     # c = np.zeros(1).reshape(1, 1)
    #     # exMat2 = np.r_[T, [c]]
    #
    #     exMatTmp = np.c_[R, T]
    #     b = np.zeros(4).reshape(1, 4)
    #     exMat = np.r_[exMatTmp, [b]]
    #
    #     n = worldPoints.shape(0)
    #     c = np.zeros(n).reshape(1, n)
    #     worldPointsAd = np.r_[worldPoints, [c]]
    #     imgPoints = np.dot(exMat, worldPointsAd)
    #     pixelPoints = np.zeros(2*n).reshape(2, n)
    #     for k in range(n):
    #         pixelPoints[:, k] = np.dot(intrinMatTransF, imgPoints[:, k])/imgPoints[3, k]
    #     return pixelPoints

    def f(self, x, input):
        xa = np.zeros((self.n, 1))
        xa[0] = x[0] # z v
        xa[1] = x[1]+x[0] # z
        return xa

    def h(self, m, x, intrinMat_left, intrinMat_right, R_left, R_right, T_left, T_right):
        # # za = np.zeros((2*2*m, 1))
        # za = []
        # pixelPoints_left = self.world2pixel(self, intrinMat_left, R_left, T_left)
        # pixelPoints_right = self.world2pixel(self, intrinMat_right, R_right, T_right)
        # za.append()
        return za

    def ut_f(self, Xsigma, Wm, Wc, finput):
        LL = Xsigma.shape[1]
        Xmeans = np.zeros((self.n, 1))
        Xsigma_pre = np.zeros((self.n, LL))
        for k in range(LL):
            Xsigma_pre[:, k] = self.f(Xsigma[:, k], finput)
            Xmeans = Xmeans + Wm[0, k] * Xsigma_pre[:, k].reshape((self.n, 1))
        Xdiv = Xsigma_pre - np.tile(Xmeans, (1, LL))
        P = np.dot(np.dot(Xdiv, np.diag(Wc.reshape((LL,)))), Xdiv.T) + self.Q

        return Xmeans, Xsigma_pre, P, Xdiv

    def ut_h(self, Xsigma, Wm, Wc, m, hinput):
        LL = Xsigma.shape[1]
        Xmeans = np.zeros((m, 1))
        Xsigma_pre = np.zeros((m, LL))
        for k in range(LL):
            Xsigma_pre[:, k] = self.h(m, Xsigma[:, k], hinput)
            Xmeans = Xmeans + Wm[0, k] * Xsigma_pre[:, k].reshape((m, 1))
        Xdiv = Xsigma_pre - np.tile(Xmeans, (1, LL))
        P = np.dot(np.dot(Xdiv, np.diag(Wc.reshape((LL,)))), Xdiv.T) + self.R

        return Xmeans, Xsigma_pre, P, Xdiv

    def output(self, alpha_msm, x0, Q, R, P):

        z = np.array(alpha_msm).reshape((3, 1))

        self.GetParameter(Q, R, P, x0)

        l = self.n
        m = z.shape[0]
        alpha = 2
        ki = 3 - l
        beta = 2
        lamb = alpha ** 2 * (l + ki) - l
        c = l + lamb
        Wm = np.concatenate((np.array(lamb / c).reshape((-1, 1)), 0.5 / c + np.zeros((1, 2 * l))), axis=1)
        Wc = Wm.copy()
        Wc[0][0] = Wc[0][0] + (1 - alpha ** 2 + beta)
        c = math.sqrt(c)

        Xsigmaset = self.sigmas(x0, c)
        X1means, X1, P1, X2 = self.ut_f(Xsigmaset, Wm, Wc)
        Zpre, Z1, Pzz, Z2 = self.ut_h(X1, Wm, Wc, m)

        Pxz = np.dot(np.dot(X2, np.diag(Wc.reshape((self.n * 2 + 1,)))), Z2.T)
        K = np.dot(Pxz, np.linalg.inv(Pzz))

        X = (X1means + np.dot(K, z - Zpre)).reshape((-1,))
        self.P = P1 - np.dot(K, Pxz.T)

        return X, self.P
