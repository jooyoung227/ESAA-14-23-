# 0511 수상작 리뷰

---

### 주제

와인 품질 분류를 위한 EDA 및 시각화, 그리고 RandomForest 기반 품질 예측 모델링

### 데이터

`train.csv`, `test.csv`

### 변수

`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`,  `type`

### 데이터 흐름

1. **데이터 로드 및 확인**: `train`, `test`, `sample_submission` 데이터를 불러오고 `info()`, `describe()`로 데이터 구조와 결측치를 확인함.
2. **EDA 수행**: `countplot`으로 타깃 변수 `quality`의 분포를 확인하고, 시각화를 통해 변수별 분포와 품질 등급 간 차이를 파악함.
3. **상관관계 분석**: `corr()`와 `heatmap`을 이용해 변수 간 관계와 `quality`에 영향을 줄 수 있는 주요 변수를 확인함.
4. **전처리 진행**: 불필요한 `index` 컬럼을 제거하고, `type` 변수를 숫자로 인코딩한 뒤 수치형 변수들을 스케일링함.
5. **모델 학습 및 평가**: `RandomForestClassifier`를 학습시키고 `accuracy_score`로 모델 성능을 평가함.
6. **최종 예측 및 제출 파일 생성**: 학습된 모델로 `test` 데이터를 예측하고 `submission.csv` 파일을 생성함.

### 주요 코드

상관관계 분석 코드

```jsx
plt.figure(figsize=(18,8))
corr= train.corr()
sns.heatmap(corr, annot=True, square=False, vmin=-.6, vmax=1.0);
```

데이터 전처리 코드

```jsx
#Standardscaler
ss= StandardScaler()
train[numerical_columns] = ss.fit_transform(train[numerical_columns])

#factorize
train['type'] = pd.factorize(train['type'])[0]

train.head(3)
```

RandomForest 모델 학습 코드

```jsx
#RandomForest
rf= RandomForestClassifier()
rf.fit(X_train,y_train)
Model(rf)
```

### 배운 점

`quality`처럼 등급별 개수 차이가 있는 타깃 변수는 모델링 전에 `value_counts()`와 `countplot`으로 분포를 먼저 확인해야 한다는 점을 배울 수 있다.

또한 `type`처럼 문자형으로 되어 있는 변수는 RandomForest 모델에 그대로 넣을 수 없기 때문에 `pd.factorize()`로 숫자형 변환이 필요하다는 점을 알 수 있다.

`StandardScaler`를 사용해 `alcohol`, `density`, `residual sugar`처럼 단위와 범위가 다른 수치형 변수들을 같은 기준으로 맞추는 전처리 과정도 확인할 수 있다.

마지막으로 `corr()`와 `heatmap`을 통해 모델을 만들기 전에 어떤 변수들이 서로 강하게 관련되어 있는지 시각적으로 점검하는 과정의 중요성을 배울 수 있다.