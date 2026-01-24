# ==================================================
# File: nvidea_core_llm_guardrail_training.py
# Author: 성진
# Date: 2026-01-24
# Description: NVIDIA 기반 LangChain Guardrailing 임베딩을 활용한 분류 모델 학습.
#              PCA/t-SNE 시각화 후 신경망과 로지스틱 회귀로 좋은/나쁜 응답을 구분한다.
# Usage:★★ - 단독 실행 가능
# ==================================================

## ---- 라이브러리 임포트 ----
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

## ---- 데이터 결합 및 라벨링 ----
embeddings = np.vstack([good_embeds, poor_embeds])
labels = np.array([0]*len(good_embeds) + [1]*len(poor_embeds))

## ---- PCA 변환 ----
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

## ---- t-SNE 변환 ----
tsne = TSNE(n_components=2, random_state=0)
embeddings_tsne = tsne.fit_transform(embeddings)

## ---- 시각화 ----
plt.figure(figsize=(12, 6))

# PCA 시각화
plt.subplot(1, 2, 1)
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels, cmap='viridis')
plt.title("PCA 임베딩 시각화")
plt.xlabel("PCA 성분 1")
plt.ylabel("PCA 성분 2")
plt.colorbar(label='그룹')

# t-SNE 시각화
plt.subplot(1, 2, 2)
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis')
plt.title("t-SNE 임베딩 시각화")
plt.xlabel("t-SNE 성분 1")
plt.ylabel("t-SNE 성분 2")
plt.colorbar(label='그룹')

plt.show()

## ---- Keras 임포트 ----
with Timer():
    print("Keras 최초 임포트")
    import keras
    from keras import layers

## ---- 신경망 학습 함수 정의 ----
def train_model_neural_network(class0, class1):
    '''
    간단한 신경망 학습 루프.
    얕은 네트워크 구조로 빠르게 수렴하며,
    좋은/나쁜 임베딩을 구분하는 이진 분류 모델을 학습한다.
    '''
    model = keras.Sequential([
        layers.Dense(64, activation='tanh'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 1),
        loss = [keras.losses.BinaryCrossentropy(from_logits=False)],
        metrics = [keras.metrics.BinaryAccuracy()],
    )
    reps_per_batch = 64*5   # 데이터 반복으로 epoch 효과 증가
    epochs = 2              # 2 epoch만으로도 충분히 수렴
    x = np.array((class0 + class1) * reps_per_batch)
    y = np.array(([0]*len(class0) + [1]*len(class1)) * reps_per_batch)
    model.fit(x, y, epochs=epochs, batch_size=64, validation_split=.5)
    return model

## ---- 신경망 학습 실행 ----
with Timer():
    model1 = train_model_neural_network(poor_embeds, good_embeds)

## ---- 로지스틱 회귀 학습 함수 정의 ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_logistic_regression(class0, class1):
    '''
    로지스틱 회귀 학습.
    수학적으로 최적화된 폐쇄형 알고리즘을 사용하여
    좋은/나쁜 임베딩을 구분한다.
    '''
    x = class0 + class1
    y = [0] * len(class0) + [1] * len(class1)
    x0, x1, y0, y1 = train_test_split(x, y, test_size=0.5, random_state=42)
    model = LogisticRegression()
    model.fit(x0, y0)
    print(np.array(x0).shape)
    print("훈련 정확도:", model.score(x0, y0))
    print("테스트 정확도:", model.score(x1, y1))
    return model

## ---- 로지스틱 회귀 학습 실행 ----
with Timer():
    model2 = train_logistic_regression(poor_embeds, good_embeds)
