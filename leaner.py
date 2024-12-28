import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100개의 랜덤 X값 생성
y = 4 + 3 * X + np.random.randn(100, 1)  # 선형 방정식 y = 4 + 3x + noise

# 데이터 시각화
plt.scatter(X, y, color='blue', alpha=0.5)
plt.title("Generated Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# 2. 선형회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 학습된 모델의 절편과 기울기 확인
print("절편 (Intercept):", model.intercept_)
print("기울기 (Slope):", model.coef_)

# 3. 예측 및 시각화 
X_new = np.array([[0], [2]])    
y_new = model.predict(X_new)
print("예측값:", y_new)

# 4. 예측 결과 시각화
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_new, y_new, color='red', linewidth=3)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()  

# 4. 특정 값 예측
test_X = np.array([[1.5], [3.0]])  # X=1.5, X=3.0
test_y_pred = model.predict(test_X)

print("X=1.5일 때 예측 값:", test_y_pred[0])
print("X=3.0일 때 예측 값:", test_y_pred[1])
