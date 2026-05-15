# 0515 수상작 리뷰

---

### 주제

LLM을 활용한 문장 순서 예측 문제를 해결

### 데이터

train.csv, test.csv

### 변수

| `MODEL_REPO` | Hugging Face에서 불러올 모델 이름 |
| --- | --- |
| `MODEL_NAME` | 모델 저장 및 결과 파일명에 사용할 모델 이름 |
| `MODEL_DIR` | 모델이 저장되는 로컬 경로 |
| `DEVICE` | GPU 또는 CPU 사용 여부 |
| `seed` | 실험 재현성을 위한 랜덤 시드 |
| `MAX_SEQ_LEN` | 모델 입력으로 사용할 최대 토큰 길이 |

### 데이터 흐름

1. 라이브러리 환경 설정, 모델 다운 및 로드
2. 정답 문장 순서 만들기
    
    4개의 정답 문장 순서를 기준으로 문장을 재배열하여 하나의 자연스러운 문장으로 제작
    
    `→` 기호로 연결
    `correct_order`라는 새로운 학습용 텍스트 컬럼이 생성
    
3. train/valid 분리 및 토큰화
전체 학습 데이터를 train과 valid로 나누고, Hugging Face `Dataset` 형태로 변환
    
    토크나이저를 이용해 문장을 모델이 이해할 수 있는 token ID 형태로 변
    
4. Fine-tuning
    
    모델을 fine-tuning하기 전 원래 모델이 문장 순서를 얼마나 잘 맞히는지 PPL을 계산하여 확인
    
    정답 순서로 연결된 문장들을 이용해 LLM을 fine-tuning
    
    fine-tuning이 끝난 뒤 다시 train과 valid 데이터에 대해 PPL 기반 문장 순서 예측을 수행
    
5. QDoRA 방식의 LoRA fine-tuning
    
    4-bit로 양자화해서 GPU 메모리 사용량을 줄이는 설정
    
    큰 LLM은 그대로 학습하기 어렵기 때문에, LoRA 또는 QDoRA를 이용해 일부 파라미터만 효율적으로 학습
    
6. 여러 모델 결과 결합
    
    여러 LLM에서 나온 PPL 결과를 불러와 하나의 데이터프레임으로 합침
    
    FLAML 기반 앙상블 랭킹 모델 학습을 하여 문제를 단순 분류가 아니라 랭킹 문제로 전환
    

### 주요 코드

정답 순서대로 문장 연결하기

```jsx
def correct_order_sentences(x):
    sentences = x[['sentence_0', 'sentence_1', 'sentence_2', 'sentence_3']].tolist()
    orders = x[['answer_0', 'answer_1', 'answer_2', 'answer_3']].tolist()
    cor_ord_sent = [sentences[x] for x in orders]
    cor_ord_sent = '→'.join(cor_ord_sent)
    return cor_ord_sent
```

데이터 토큰화

```jsx
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
        max_length=MAX_SEQ_LEN
    )
    return tokenized_output
```

Fine-tuning 설정

```jsx
training_args = TrainingArguments(
    output_dir=checkpoint_dir_path,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
```

### 배울 점

이 과제를 통해 LLM을 실제 문제에 맞게 활용하는 전체 흐름을 배울 수 있었다. 특히 4개의 문장으로 가능한 모든 순열을 만들고, 각 순열의 PPL을 계산해 가장 자연스러운 문장 순서를 고르는 방식이 인상적이었다. 이를 통해 LLM이 문장 생성뿐 아니라 문장의 자연스러움과 연결성을 평가하는 데에도 활용될 수 있음을 알게 되었다.

또한 fine-tuning 과정에서 train/valid 데이터를 나누고, 검증 손실이 가장 낮은 step을 기준으로 재학습하는 흐름을 이해할 수 있었다. 더불어 4-bit 양자화, LoRA, QDoRA, gradient accumulation 등을 통해 대형 LLM 학습에서는 GPU 메모리 관리와 학습 효율화 전략이 매우 중요하다는 점을 배웠다. 마지막으로 여러 모델의 PPL 결과를 앙상블하고 FLAML 기반 랭킹 모델로 최종 예측을 수행하면서, 단일 모델보다 여러 모델의 판단을 결합하는 방식이 더 안정적인 예측에 도움이 된다는 것을 알게 되었다.