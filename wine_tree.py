from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_text

# 와인 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정트리 모델 생성
tree = DecisionTreeClassifier(
    max_depth=3,           # 트리의 깊이를 3으로 제한 (시각화를 위해)
    random_state=42
)

# 모델 학습
tree.fit(X_train, y_train)

# 결정트리 시각화
plt.figure(figsize=(20,10))
plot_tree(tree, 
    feature_names=wine.feature_names,    # 특성 이름
    class_names=wine.target_names,       # 클래스 이름
    filled=True,                         # 노드 색칠
    rounded=True,                        # 모서리 둥글게
    fontsize=10                          # 글자 크기
)
plt.show()

# 모델 성능 출력
print(f"훈련 세트 정확도: {tree.score(X_train, y_train):.3f}")
print(f"테스트 세트 정확도: {tree.score(X_test, y_test):.3f}")

# 특성 중요도 시각화
#importances = tree.feature_importances_
#feature_names = wine.feature_names

#plt.figure(figsize=(10,6))
#plt.bar(feature_names, importances)
#plt.xticks(rotation=45, ha='right')
#plt.title("Feature Importances in Wine Classification")
#plt.tight_layout()
#plt.show() 

# 텍스트 형태로 트리 출력
tree_text = export_text(tree,
    feature_names=wine.feature_names,    # 특성 이름
    show_weights=True,                   # 각 노드의 샘플 수 표시
    spacing=3                            # 들여쓰기 간격
)

print("\n=== 결정 트리 텍스트 표현 ===")
print(tree_text)