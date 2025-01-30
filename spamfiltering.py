import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 로드 (예시: SMS Spam Collection 데이터셋)
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]  # 'v1'은 라벨, 'v2'는 메시지 텍스트

# 데이터 전처리
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # 'ham'은 0, 'spam'은 1로 변환

# 훈련 데이터와 테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)

# 텍스트를 피처 벡터로 변환 (단어 빈도 수 벡터화)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 나이브 베이즈 모델 학습
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# 예측
y_pred = nb_model.predict(X_test_vectorized)

# 정확도와 분류 리포트 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve 시각화
fpr, tpr, thresholds = roc_curve(y_test, nb_model.predict_proba(X_test_vectorized)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# 정밀도-재현율 곡선 시각화
precision, recall, _ = precision_recall_curve(y_test, nb_model.predict_proba(X_test_vectorized)[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='b')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
