# 0522 수상작 리뷰

---

### 주제

**모기 비행 궤적 예측 AI 경진대회**

물리 기반 후보군과 딥러닝 모델을 결합하여 모기/비행체의 다음 위치를 예측하는 문제를 해결

### 데이터

`train`, `test`, `train_labels.csv`, `sample_submission.csv`

### 변수

`DATA_ROOT`학습 데이터와 테스트 데이터가 저장된 기본 경로

`WORK_DIR`모델 학습 결과, OOF score, report 등을 저장하는 작업 폴더

`R_HIT`예측 위치가 정답과 1cm 이내이면 hit으로 판단하는 기준값

`SEQ_FEATURE_NAMES`trajectory의 최근 움직임을 설명하는 시계열 feature 이름

`CAND_FEATURE_NAMES`각 후보 위치가 어떤 물리적 의미를 가지는지 설명하는 후보 feature 이름

`CANDIDATES`기본 위치, 가속도, Frenet frame, turn, jerk, latency 등을 반영한 물리 기반 후보군

`cluster_labels / score_bank`selector 모델이 각 후보에 부여한 점수 및 OOF score 저장 파일

`cap`boundary correction에서 허용하는 최대 보정 크기

### 데이터 흐름

1. 데이터 경로 설정

로컬 또는 Colab 환경에서 `train`, `test`, `train_labels.csv`, `sample_submission.csv`가 존재하는 경로를 찾아 `DATA_ROOT`로 지정한다. 이후 결과 저장을 위한 `WORK_DIR`을 생성한다.

1. 물리 기반 후보군 생성

마지막 관측 위치, 최근 속도, 가속도, 방향 전환, jerk, latency 등을 이용해 여러 후보 위치를 만든다. 이 후보들은 단순 좌표 예측값이 아니라, 실제 움직임의 물리적 가능성을 반영한 위치들이다.

1. 후보별 feature 생성

각 후보가 최근 이동 방향과 얼마나 일치하는지, 속도 대비 얼마나 이동했는지, 수직 방향 움직임이 있는지 등을 feature로 만든다. 또한 trajectory 자체에서도 속도, 가속도, 곡률, 방향 변화 등의 sequence feature를 추출한다.

1. Attn-GRU Candidate Selector 학습

GRU 기반 모델이 trajectory 정보를 요약하고, attention 구조를 통해 후보별 점수를 계산한다. 모델은 직접 좌표를 예측하지 않고, 후보들 중 어떤 후보를 믿을지를 학습한다.

1. OOF score bank 저장

selector가 각 fold에서 예측한 후보 점수를 저장한다. 이 score bank는 이후 boundary correction 단계에서 사용된다.

1. Boundary Residual MLP 보정

selector가 선택한 후보가 1cm hit boundary 근처에서 살짝 벗어난 경우를 보정하기 위해 작은 MLP 모델을 사용한다. 이때 보정량은 `cap`으로 제한되어, 모델이 큰 좌표 이동을 새로 학습하지 못하게 한다.

1. 성능 요약

selector의 soft hit, gate hit, boundary correction 이후 hit, oracle hit 등을 report에서 불러와 성능을 비교한다.

### 주요코드

selector 학습 실행

```jsx
selector_out = WORK_DIR / 'outputs/selector_1fold_inline_attn_gru'

call_main(SELECTOR_MAIN, [
    '--root', DATA_ROOT,
    '--out-dir', selector_out,
    '--models', 'attn_gru',
    '--folds', 5,
    '--fold-limit', 1,
    '--pre-epochs', 1,
    '--fine-epochs', 1,
    '--device', 'auto'
])
```

Boundary Residual MLP 실행

```jsx
score_bank = selector_out / 'oof_selector_scores.npz'

boundary_out = WORK_DIR / 'outputs/boundary_1fold_inline_resmlp'

call_main(BOUNDARY_MAIN, [
    '--root', DATA_ROOT,
    '--out-dir', boundary_out,
    '--fold', 0,
    '--folds', 5,
    '--score-bank', score_bank,
    '--cap', 0.006,
    '--save-val-pred'
])
```

성능 요약

```jsx
summary = {
    'selector_soft_hit': selector_report['model_oof']['attn_gru']['soft']['metrics']['hit'],
    'selector_gate_hit': selector_report['model_oof']['attn_gru']['argmax_soft_gate']['metrics']['hit'],
    'boundary_soft_hit': boundary_report['soft']['metrics']['hit'],
    'boundary_gate_hit': boundary_report['gate']['metrics']['hit'],
    'boundary_argmax_hit': boundary_report['argmax']['hit'],
    'boundary_oracle_hit': boundary_report['candidate_oracle']['hit'],
}
```

### 배울 점

이 코드를 통해 단순히 좌표를 직접 예측하는 방식보다, 먼저 물리적으로 가능한 후보군을 만들고 그중에서 모델이 선택하도록 하는 방식이 더 안정적일 수 있음을 배울 수 있었다. 특히 마지막 위치, 속도, 가속도뿐 아니라 Frenet frame, turn, jerk, latency까지 반영하여 후보를 구성한 점이 인상적이었다.

또한 GRU와 attention을 사용해 trajectory의 흐름을 요약하고, 후보별 feature와 결합하여 최종 후보 점수를 계산하는 구조를 이해할 수 있었다. 모델이 모든 것을 직접 맞히는 것이 아니라, 물리 후보 중 어떤 것을 신뢰할지 학습한다는 점에서 과적합을 줄이려는 설계가 드러난다.

마지막으로 boundary correction 단계에서는 큰 보정을 허용하지 않고, 1cm 기준 근처에서만 작은 residual을 보정하도록 제한했다. 이를 통해 성능을 높이면서도 모델이 노이즈나 환경 차이를 과하게 학습하지 않도록 조절하는 전략을 배울 수 있었다. 전체적으로 이 코드는 물리 기반 feature engineering, 후보 선택 모델, OOF score 활용, residual correction을 하나의 파이프라인으로 연결한 고급 예측 구조라고 볼 수 있다.