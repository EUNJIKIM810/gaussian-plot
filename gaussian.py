import numpy as np
import matplotlib.pyplot as plt

# 정규분포 함수
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# x 범위 설정
x = np.linspace(-10, 10, 1000)

# μ와 σ 값 변경에 따른 그래프
mu_values = [0, 0, -2]
sigma_values = [1, 2, 1]

# 그래프 그리기
plt.figure(figsize=(10, 6))
for mu, sigma in zip(mu_values, sigma_values):
    y = gaussian(x, mu, sigma)
    label = f"μ={mu}, σ={sigma}"
    plt.plot(x, y, label=label)

plt.title("Effect of μ and σ on Gaussian Distribution", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.4)
plt.show()
