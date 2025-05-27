# Pattern Recognition Repository
```
pattern-recognition/
│
├─ data/
│   ├─ train.csv
│   ├─ test.csv
│   └─ variable_information.csv
│
├─ data_preprocessing/
│   ├─ mean_impute_std_scale_onehot/
│   │   ├─ main_mean_impute_std_scale_onehot.ipynb  ← 메인 실행 파일
│   │   └─ result/
│   │       └─ preprocessed_train.csv
│
├─ data_analysis/
│   └─ ...  ← 분석 관련 노트북·스크립트
│
├─ model/
│   ├─ baseline_model/
│   │   ├─ main_baseline.ipynb
│   │   └─ result.csv  ← baseline 결과물
│   │
│   ├─ catboost/
│   │   ├─ main_catboost.ipynb  ← CatBoost 모델 실행 파일
│   │   └─ result.csv  ← CatBoost 결과물
│   │
│   ├─ softvoting_catboost_gbm/
│   │   ├─ main_softvoting_catboost_gbm.ipynb
│   │   └─ result.csv
│   │
│   └─ ...  ← 기타 모델
│
└─ README.md
```
---

## 📁 디렉터리 & 파일명 규칙

### 1. 전처리 (data_preprocessing/)
- 메인 폴더/스크립트:
  data_preprocessing/<description>/main_<description>.ipynb  
  예: mean_impute_std_scale_onehot → main_mean_impute_std_scale_onehot.ipynb
- 결과물 폴더:
  data_preprocessing/<description>/result/
  결과물이 많으면 내부에 CSV, PNG 등 저장

### 2. 모델 (model/)
- 서브폴더 구조:
```
  model/
  └─ <model_name>/
      ├─ main_<model_name>.ipynb
      └─ result/
          └─ <model_name>_metrics.csv
```
- 파일명 컨벤션 (Notebooks)
  1. 소문자 + snake_case  
  2. 접두사 main_ = “메인 실행 파일”  
  3. 형식:
     main_<기법>[_<mod1>][_ <mod2>...].ipynb
     - <기법>: 주 모델 이름 (e.g. catboost, xgboost, lightgbm, gaussian_nb)
     - <mod>: 하이퍼파라미터 변경, 앙상블 기법 등 추가 정보
  4. 앙상블은 별도 폴더 생성 또는 파일명에 반영:
     예: softvoting_catboost_gbm → main_softvoting_catboost_gbm.ipynb

- 결과 폴더:
  model/<model_name>/result/
  성능 지표(CSV), 시각화(PNG) 등 저장

### 3. 앙상블 / 복합 모델
- 서브폴더를 만들지 않을 경우:
  model/softvoting_catboost_gbm.ipynb
- 결과는 model/result/ 또는 model/softvoting_catboost_gbm/result/에 저장 가능

---

## 📝 요약
1. 소문자 + snake_case  
2. 접두사 main_ = “메인 실행 파일”  
3. [기법]_[변경사항...] 순서로 명시  
4. 각 단계별 result/ 폴더에 출력물 저장

---
