import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('/home/kanghosung/hw1_patt/data/train.csv')


df['target'] = (df['shares'] > 1400).astype(int)
df = df.drop(columns=['id', 'shares', 'y'], errors='ignore')

# 범주형 및 수치형 변수 구분
categorical_cols = ['data_channel', 'weekday']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['target']]

# 수치형 파이프라인: 평균 대체 + 표준화
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 범주형 파이프라인: 최빈값 대체 + 원핫 인코딩
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 컬럼 전처리기 정의
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


X = df.drop(columns=['target'])
y = df['target'].values


X_processed = preprocessor.fit_transform(X)


feature_names_num = numerical_cols
feature_names_cat = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
feature_names = np.concatenate([feature_names_num, feature_names_cat])


df_processed = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed, columns=feature_names)
df_processed['target'] = y


df_processed.to_csv('/home/kanghosung/hw1_patt/data_preprocessing/result/trial1_train.csv', index=False)
