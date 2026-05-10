# 수상작 리뷰

## 주제

난임 환자 대상 임신 성공 여부 예측 AI

[https://dacon.io/competitions/official/236452/codeshare/12308](https://dacon.io/competitions/official/236452/codeshare/12308)

## 데이터

`train.csv`

`ID`, `시술 당시 나이`, `시술 시기 코드`, `시술 유형`, `특정 시술 유형`, `배란 자극 여부`, `배란 유도 유형`, `단일 배아 이식 여부`, `착상 전 유전 검사 사용 여부`, `착상 전 유전 진단 사용 여부`, `남성 주 불임 원인`, `남성 부 불임 원인`, `여성 주 불임 원인`, `여성 부 불임 원인`, `부부 주 불임 원인`, `부부 부 불임 원인`, `불명확 불임 원인`, `불임 원인 - 여성 요인`, `불임 원인 - 난관 질환`, `불임 원인 - 남성 요인`, `불임 원인 - 배란 장애`, `불임 원인 - 자궁경부 문제`, `불임 원인 - 자궁내막증`, `불임 원인 - 정자 농도`, `불임 원인 - 정자 면역학적 요인`, `불임 원인 - 정자 운동성`, `불임 원인 - 정자 형태`, `배아 생성 주요 이유`, `난자 출처`, `정자 출처`, `난자 기증자 나이`, `정자 기증자 나이`, `동결 배아 사용 여부`, `신선 배아 사용 여부`, `기증 배아 사용 여부`, `대리모 여부`, `PGD 시술 여부`, `PGS 시술 여부`, `총 시술 횟수`, `클리닉 내 총 시술 횟수`, `IVF 시술 횟수`, `DI 시술 횟수`, `총 임신 횟수`, `IVF 임신 횟수`, `DI 임신 횟수`, `총 출산 횟수`, `IVF 출산 횟수`, `DI 출산 횟수`, `임신 시도 또는 마지막 임신 경과 연수`, `난자 채취 경과일`, `난자 해동 경과일`, `난자 혼합 경과일`, `배아 이식 경과일`, `배아 해동 경과일`, `총 생성 배아 수`, `미세주입된 난자 수`, `미세주입에서 생성된 배아 수`, `이식된 배아 수`, `미세주입 배아 이식 수`, `저장된 배아 수`, `미세주입 후 저장된 배아 수`, `해동된 배아 수`, `해동 난자 수`, `수집된 신선 난자 수`, `저장된 신선 난자 수`, `혼합된 난자 수`, `파트너 정자와 혼합된 난자 수`, `기증자 정자와 혼합된 난자 수`, `임신 성공 여부`

`test.csv`

`ID`, `시술 당시 나이`, `시술 시기 코드`, `시술 유형`, `특정 시술 유형`, `배란 자극 여부`, `배란 유도 유형`, `단일 배아 이식 여부`, `착상 전 유전 검사 사용 여부`, `착상 전 유전 진단 사용 여부`, `남성 주 불임 원인`, `남성 부 불임 원인`, `여성 주 불임 원인`, `여성 부 불임 원인`, `부부 주 불임 원인`, `부부 부 불임 원인`, `불명확 불임 원인`, `불임 원인 - 여성 요인`, `불임 원인 - 난관 질환`, `불임 원인 - 남성 요인`, `불임 원인 - 배란 장애`, `불임 원인 - 자궁경부 문제`, `불임 원인 - 자궁내막증`, `불임 원인 - 정자 농도`, `불임 원인 - 정자 면역학적 요인`, `불임 원인 - 정자 운동성`, `불임 원인 - 정자 형태`, `배아 생성 주요 이유`, `난자 출처`, `정자 출처`, `난자 기증자 나이`, `정자 기증자 나이`, `동결 배아 사용 여부`, `신선 배아 사용 여부`, `기증 배아 사용 여부`, `대리모 여부`, `PGD 시술 여부`, `PGS 시술 여부`, `총 시술 횟수`, `클리닉 내 총 시술 횟수`, `IVF 시술 횟수`, `DI 시술 횟수`, `총 임신 횟수`, `IVF 임신 횟수`, `DI 임신 횟수`, `총 출산 횟수`, `IVF 출산 횟수`, `DI 출산 횟수`, `임신 시도 또는 마지막 임신 경과 연수`, `난자 채취 경과일`, `난자 해동 경과일`, `난자 혼합 경과일`, `배아 이식 경과일`, `배아 해동 경과일`, `총 생성 배아 수`, `미세주입된 난자 수`, `미세주입에서 생성된 배아 수`, `이식된 배아 수`, `미세주입 배아 이식 수`, `저장된 배아 수`, `미세주입 후 저장된 배아 수`, `해동된 배아 수`, `해동 난자 수`, `수집된 신선 난자 수`, `저장된 신선 난자 수`, `혼합된 난자 수`, `파트너 정자와 혼합된 난자 수`, `기증자 정자와 혼합된 난자 수`

`sample_submission.csv`

 `ID`, `probability`

## 코드흐름

1. 라이브러리 불러오기
2. 데이터 변환 함수 정의 
    
    `number_mapping()` 함수를 통해 `0회`, `1회`, `6회 이상`처럼 문자로 표현된 시술 횟수 변수를 숫자형으로 변환
    범주형 변수와 수치형 변수를 구분
    CatBoost 모델에 맞게 컬럼 타입을 변환하는 함수를 정의
    
3. 결측치 처리 전 전처리
    
    `pre_mice()`에서 `배아 해동 여부`, `배란 유도_배란 자극` 같은 파생변수를 생성
    `pp_fillna_di()`에서는 시술 유형이 DI인 경우, IVF나 배아 관련 변수의 결측치를 0으로 채움
    
4. MICE 결측치 보간
    
    배아 수, 난자 수, 경과일 등 결측치가 많은 주요 변수들을 대상으로 MICE 모델을 사용해 결측값을 예측하여 채움
    
    학습 데이터로 MICE 모델을 학습하거나, 저장된 MICE 모델을 불러와 train/test 데이터에 동일하게 적용
    
    ```jsx
    def mice_process(train, test, target_columns, mdl):
        train = apply_mice(train, target_columns, mdl, True)
        test = apply_mice(test, target_columns, mdl, False)
    
        return train, test
    ```
    
    ```jsx
    def apply_mice(df, target_columns, mdl, isTraining):
        df_subset = df[target_columns].copy()
    
        if isTraining:
            imputed_df = mdl.complete_data()
        else:
            imputed_df = mdl.impute_new_data(df_subset).complete_data()
    
        df[target_columns] = imputed_df[target_columns]
    
        return df
    ```
    
5. 파생변수 생성
`pp_ratio()` 함수를 통해 기존 변수들을 조합하여 새로운 비율 변수 생성

6. 불필요한 컬럼 제거 및 타입 변환
    
    `ID`처럼 예측에 직접 필요하지 않은 컬럼과 일부 중복되거나 영향이 낮은 컬럼을 제거
    범주형 변수는 category/string 형태로, 수치형 변수는 numeric 형태로 변환
    
7. train/test 데이터 전처리 적용
    
    `train.csv`와 `test.csv`를 불러온 뒤, 앞에서 정의한 전처리 과정을 동일하게 적용
    
8. CatBoost 모델 학습
    
    Optuna를 통해 찾은 하이퍼파라미터를 CatBoostClassifier에 적용
    
    `StratifiedKFold`를 사용해 임신 성공 여부의 클래스 비율이 유지되도록 데이터를 25개 fold로 나누고, 각 fold마다 학습과 검증을 반복
    
    ```jsx
    skf = StratifiedKFold(
        n_splits=k_fold,
        shuffle=True,
        random_state=random_state
    )
    
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train, df_train["임신 성공 여부"])
    ):
        train_data, val_data = df_train.iloc[train_idx], df_train.iloc[val_idx]
    
        x_train = train_data.drop('임신 성공 여부', axis=1)
        y_train = train_data["임신 성공 여부"]
    
        x_val = val_data.drop('임신 성공 여부', axis=1)
        y_val = val_data["임신 성공 여부"]
    
        train_pool = Pool(x_train, label=y_train, cat_features=category_columns)
        eval_pool = Pool(x_val, label=y_val, cat_features=category_columns)
    
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=100,
            early_stopping_rounds=250
        )
    ```
    
9. 모델 성능 평가
    
    각 fold에서 검증 데이터에 대해 예측을 수행하고, confusion matrix, accuracy, precision, recall, f1-score, roc-auc를 계산
    
10. test 데이터 예측 및 제출 파일 생성
각 fold에서 학습된 모델로 `test.csv`의 임신 성공 확률을 예측

## 새롭게 알게 된 내용 / 어려운 내용 / 배울 점

- 이 코드를 통해 데이터 분석에서 모델 성능은 단순히 알고리즘 선택만으로 결정되는 것이 아니라, 변수 전처리와 파생변수 생성 과정에 크게 영향을 받는다는 점을 알게 되었다.
- 기존 변수를 그대로 사용하는 것보다 `IVF 임신률`, `DI 임신률`, `IVF 출산률`, `미세주입 배아 생성률`처럼 비율 형태의 파생변수를 생성하면 데이터 안에 숨어 있는 의미를 더 잘 반영할 수 있다는 점을 배웠다.
- 이 파일을 통해 배울 점은 데이터 분석에서 전처리와 feature engineering이 매우 중요하다는 것이다. 좋은 모델을 사용하는 것도 중요하지만, 모델에 넣기 전 데이터를 어떻게 정리하고 어떤 변수를 새로 만들 것인지가 예측 성능에 큰 영향을 준다는 점을 배울 수 있다. 또한 train 데이터와 test 데이터에 같은 전처리 과정을 적용해야 한다는 점, target 변수인 `임신 성공 여부`는 입력 변수에서 제외하고 따로 분리해야 한다는 점도 중요하다.
