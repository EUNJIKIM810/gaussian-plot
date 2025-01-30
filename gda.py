import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 2개의 클래스 데이터 생성 (각 클래스는 2D 가우시안 분포를 따릅니다)
np.random.seed(0)

# 클래스 0: 평균 [2, 2], 공분산 행렬 [[1, 0.5], [0.5, 1]]
mu_0 = np.array([2, 2])
cov_0 = np.array([[1, 0.5], [0.5, 1]])
class_0 = np.random.multivariate_normal(mu_0, cov_0, 100)

# 클래스 1: 평균 [7, 7], 공분산 행렬 [[1, -0.5], [-0.5, 1]]
mu_1 = np.array([7, 7])
cov_1 = np.array([[1, -0.5], [-0.5, 1]])
class_1 = np.random.multivariate_normal(mu_1, cov_1, 100)

# 데이터 시각화
plt.figure(figsize=(10, 6))
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

# 평균 점 시각화
plt.scatter(mu_0[0], mu_0[1], color='blue', marker='x', s=100)
plt.scatter(mu_1[0], mu_1[1], color='red', marker='x', s=100)

plt.title("2D Gaussian Distributions for Two Classes", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# GDA를 이용해 클래스별 결정 경계 시각화
# 공통 공분산 행렬 계산 (GDA에서는 두 클래스의 공분산이 동일하다고 가정)
cov_common = (cov_0 + cov_1) / 2

# 두 클래스의 평균 벡터
mu_0 = np.array([2, 2])
mu_1 = np.array([7, 7])

# 가우시안 분포를 따르는 확률 밀도 함수 계산
x, y = np.meshgrid(np.linspace(-1, 10, 200), np.linspace(-1, 10, 200))
pos = np.dstack((x, y))

# 클래스 0과 클래스 1에 대한 가우시안 분포 확률 밀도 계산
rv_0 = multivariate_normal(mu_0, cov_common)
rv_1 = multivariate_normal(mu_1, cov_common)

# 확률 밀도 함수 값을 계산
z_0 = rv_0.pdf(pos)
z_1 = rv_1.pdf(pos)

# 결정 경계 그리기
decision_boundary = np.argmax([z_0, z_1], axis=0)

# 결정 경계 시각화
plt.contour(x, y, decision_boundary, levels=[0.5], colors='green', linewidths=2)

plt.show()
