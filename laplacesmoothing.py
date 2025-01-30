import numpy as np
import matplotlib.pyplot as plt

# 예제 단어 집합 (전체 6개 단어)
vocab = ["apple", "banana", "grape", "orange", "peach", "melon"]

# 실제 등장 횟수 (어떤 단어는 한 번도 등장하지 않음)
word_counts = {"apple": 10, "banana": 15, "grape": 0, "orange": 5, "peach": 8, "melon": 0}

# 전체 단어 수
total_words = sum(word_counts.values())

# 라플라스 스무딩 적용 전 확률 계산
no_smoothing_probs = {word: count / total_words if total_words > 0 else 0 for word, count in word_counts.items()}

# 라플라스 스무딩 적용 (α=1)
alpha = 1
total_words_smoothed = total_words + alpha * len(vocab)
laplace_smoothing_probs = {word: (count + alpha) / total_words_smoothed for word, count in word_counts.items()}

# 시각화를 위한 데이터 준비
words = list(word_counts.keys())
values_no_smoothing = list(no_smoothing_probs.values())
values_laplace_smoothing = list(laplace_smoothing_probs.values())

# 1. 라플라스 스무딩 적용 전 확률 분포 그래프
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(words, values_no_smoothing, color='red', alpha=0.7)
plt.xlabel("Words")
plt.ylabel("Probability")
plt.title("Before Laplace Smoothing (No Probability for Unseen Words)")
plt.ylim(0, max(values_laplace_smoothing) + 0.05)
for i, v in enumerate(values_no_smoothing):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=12)

# 2. 라플라스 스무딩 적용 후 확률 분포 그래프
plt.subplot(1, 2, 2)
plt.bar(words, values_laplace_smoothing, color='blue', alpha=0.7)
plt.xlabel("Words")
plt.ylabel("Probability")
plt.title("After Laplace Smoothing (All Words Have Non-Zero Probability)")
plt.ylim(0, max(values_laplace_smoothing) + 0.05)
for i, v in enumerate(values_laplace_smoothing):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=12)

plt.tight_layout()
plt.show()
