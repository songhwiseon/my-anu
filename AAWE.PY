from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np



# 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

df = pd.DataFrame(data)

# 독립 변수(X)와 종속 변수(y) 분리
X = df[['Study Hours', 'Exam Score']]
y = df['Pass']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 모델 생성
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 모델 평가
accuracy = model.score(X_test, y_test)
print(f"모델 정확도: {accuracy:.2f}")

# 결정 트리 시각화
plt.figure(figsize=(10, 8))
tree.plot_tree(model, feature_names=['Study Hours', 'Exam Score'], class_names=['Fail', 'Pass'], filled=True)
plt.show()

# 텍스트 형태로 트리 출력
tree_rules = export_text(model, feature_names=['Study Hours', 'Exam Score'])
print(tree_rules)
