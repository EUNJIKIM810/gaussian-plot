import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 2D 가우시안 분포의 평균과 공분산 행렬
mean = [0, 0]  # 평균 벡터
cov = [[1, 0], [0, 1]]  # 공분산 행렬 (항등 행렬)

# 그리드 생성
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))

# 가우시안 확률 밀도 함수 계산
rv = multivariate_normal(mean, cov)
z = rv.pdf(pos)

# 시각화
plt.contourf(x, y, z, levels=20, cmap='viridis')
plt.colorbar()
plt.title('2D Gaussian Distribution with Identity Covariance')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
