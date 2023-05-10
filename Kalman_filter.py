import cv2
import numpy as np
from matplotlib import pylab

def kalman_filter(data, Q=1e-8, R=1e-5, x0=0, P0=1):
    N = len(data)
    K = np.zeros(N)
    X = np.zeros(N)
    P = np.zeros(N)
    X[1] = x0
    P[1] = P0
    for i in range(2,N):
        K[i] = P[i - 1] / (P[i - 1] + R)
        X[i] = X[i - 1] + K[i] * (data[i] - X[i - 1])
        P[i] = P[i - 1] - K[i] * P[i - 1] + Q
    return X

"""
# def kalman_filter(n_iter, ):
# initialization of parameters
n_iter = 50
sz = (n_iter,)  # size of array
x = -0.37727  # real value
z = numpy.random.normal(x, 0.1, size=sz)  # 观测值 ,观测时存在噪声

Q = 1e-5  # process variance


def kalman_filter(z, n_iter, Q=1e-5):
    kalman = cv2.KalmanFilter(1, 1)
    kalman.transitionMatrix = np.array([[1]], np.float32)  # 转移矩阵 A
    kalman.measurementMatrix = np.array([[1]], np.float32)  # 测量矩阵    H
    kalman.measurementNoiseCov = np.array([[1]], np.float32) * 0.01  # 测量噪声        R
    kalman.processNoiseCov = np.array([[1]], np.float32) * Q  # 过程噪声 Q
    kalman.errorCovPost = np.array([[1.0]], np.float32)  # 最小均方误差 P
    xhat = np.zeros(sz)  # x 滤波估计值
    kalman.statePost = np.array([xhat[0]], np.float32)
    for k in range(1, n_iter):
        #     print(np.array([z[k]], np.float32))
        mes = np.reshape(np.array([z[k]], np.float32), (1, 1))
        # #     print(mes.shape)
        xhat[k] = kalman.predict()
        kalman.correct(np.array(mes, np.float32))


pylab.figure()
pylab.plot(z, 'k+', label='noisy measurements')  # 观测值
pylab.plot(xhat, 'b-', label='a posteri estimate')  # 滤波估计值
pylab.axhline(x, color='g', label='truth value')  # 真实值
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage')
pylab.show()

"""