# SecurityLogClassifier

## 프로젝트 소개
본 프로젝트는 대학교 "소프트웨어융합프로젝트" 과목의 일환으로 진행된 로그 분석 기반 보안 위험도 예측 AI 프로젝트입니다. Dacon에서 개최되었던 보안 위험도 예측 관련 경진대회를 참고하여, 다양한 팀들의 학습 및 예측 프로세스를 비교 분석하고, 이를 본 프로젝트에 적용하여 결과를 분석하고 모델을 개선하고자 하였습니다.

## 주요 기능
- 보안 로그 데이터 전처리
- 텍스트 기반 특징 추출
- 보안 위험도 레벨 예측 (7단계)

## 프로젝트 구조

### 1. 데이터 로드 및 EDA (SecurityLogAnalyze.ipynb)
```python
# Import Library
import pandas as pd
import numpy as np
...
```
- Google Colab 환경에서 실행
- train.csv, test.csv, sample_submission.csv 파일 로드
- 데이터 형태 및 분포 확인
- 레벨별 데이터 분포 분석

### 2. 데이터 전처리
```python
# 중복 데이터 제거
train = train.drop_duplicates(subset=['full_log'])

# 텍스트 전처리 함수
def preprocess_text(text):
    ...
```
- 중복 데이터 제거
  - full_log 기준으로 중복 제거
- 텍스트 전처리
  - IP:Port 형식 표준화
  - 특수 문자 제거
  - 불필요한 공백 제거

### 3. 특징 추출 및 데이터 분할
```python
# TF-IDF 벡터화
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3)
)
...
```
- TF-IDF 벡터화
  - 최대 10,000개 특징 추출
  - 1-3개 단어 조합(N-gram) 사용
- 데이터 분할
  - 훈련 데이터 80%
  - 검증 데이터 20%
  - 레이블 분포 유지(stratify)

### 4. 모델 학습
```python
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=7,
    ...
)
```
- LightGBM 분류기 설정
  - 다중 클래스 분류(7개 클래스)
  - 1000개 트리 생성
  - 학습률 0.05
  - 클래스 불균형 처리
- 모델 훈련
  - 검증 세트 기반 성능 모니터링
  - 50회 성능 개선 없을 시 조기 종료
  - 100회마다 평가 로그 출력

### 5. 모델 평가
```python
y_pred_val = model.predict(X_val_tfidf)
print(classification_report(y_val, y_pred_val))
```
- 검증 데이터 예측
- 분류 성능 평가

### 6. 예측 및 제출
```python
test_preds_proba = model.predict_proba(X_test_tfidf)
...
submission.to_csv('submission.csv', index=False)
```
- 테스트 데이터 예측
- 확률 임계값 기반 레벨 7 할당
- submission.csv 생성

## 설치 및 실행 방법
1. Google Colab 환경 설정
2. 필요한 데이터 파일 업로드
   - train.csv
   - test.csv
   - sample_submission.csv
3. SecurityLogAnalyze.ipynb 파일 실행

## 모델 성능
- 다중 클래스 분류 평가 지표 사용
- 조기 종료를 통한 과적합 방지
- 클래스 불균형 해결을 위한 가중치 적용

## 개선 사항
- 추가적인 텍스트 전처리 기법 적용 가능
- 다양한 특징 추출 방법 시도 가능
- 다른 분류 모델과의 앙상블 적용 가능
- 하이퍼파라미터 최적화를 통한 성능 개선 가능
- 더 세밀한 임계값 조정을 통한 레벨 7 예측 정확도 향상 가능

## 참고
https://dacon.io/competitions/official/235717/overview/description
