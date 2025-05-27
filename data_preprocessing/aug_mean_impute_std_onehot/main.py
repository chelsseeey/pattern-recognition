import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load raw data
df = pd.read_csv('/home/kanghosung/hw1_patt/data/train.csv')
df['target'] = (df['shares'] > 1400).astype(int)
df = df.drop(columns=['id', 'shares', 'y'], errors='ignore')

# 2. Identify columns
categorical_cols = ['data_channel', 'weekday']
numerical_cols = [c for c in df.columns if c not in categorical_cols + ['target']]

# 3. Compute imbalance and total synthetic needed to balance
n_neg = (df.target == 0).sum()
n_pos = (df.target == 1).sum()
n_synth_total = n_neg - n_pos

# 4. Group stats over positive class
grouped = df[df.target == 1].groupby(categorical_cols)
group_counts = grouped.size()
synth_counts = (group_counts / group_counts.sum() * n_synth_total).round().astype(int)

synth_list = []
for (ch, wd), count in synth_counts.items():
    if count <= 0:
        continue
    grp = grouped.get_group((ch, wd))[numerical_cols]
    mu = grp.mean().values
    # regularize covariance: add epsilon * identity
    cov = np.cov(grp.values, rowvar=False)
    epsilon = 1e-6 * np.trace(cov) / len(cov)  # small fraction of average variance
    cov += np.eye(len(numerical_cols)) * epsilon
    # try multivariate sampling, fallback to independent normals if it fails
    try:
        samples = np.random.multivariate_normal(mu, cov, size=count)
    except np.linalg.LinAlgError:
        var = np.diag(cov)
        samples = np.random.normal(loc=mu, scale=np.sqrt(var), size=(count, len(mu)))
    df_synth = pd.DataFrame(samples, columns=numerical_cols)
    df_synth['data_channel'] = ch
    df_synth['weekday'] = wd
    df_synth['target'] = 1
    synth_list.append(df_synth)

# combine original + synthetic
df_aug = pd.concat([df] + synth_list, ignore_index=True)

# 5. Text-feature jitter for positive samples
text_cols = [c for c in numerical_cols if any(keyword in c for keyword in ['token', 'subjectivity', 'polarity'])]
mask = df_aug.target == 1
for col in text_cols:
    scale = 0.01 * (df_aug[col].max() - df_aug[col].min())
    noise = np.random.normal(scale=scale, size=mask.sum())
    df_aug.loc[mask, col] += noise

# 6. Preprocess and save
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numerical_cols),
    ('cat', cat_pipe, categorical_cols)
])

X_aug = df_aug.drop(columns=['target'])
y_aug = df_aug['target'].values
X_proc = preprocessor.fit_transform(X_aug)
num_feats = numerical_cols
cat_feats = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_feats = np.concatenate([num_feats, cat_feats])
df_final = pd.DataFrame(X_proc.toarray() if hasattr(X_proc, 'toarray') else X_proc, columns=all_feats)
df_final['target'] = y_aug

# Save augmented processed data
df_final.to_csv('/home/kanghosung/hw1_patt/pattern-recognition/data_preprocessing/aug_mean_impute_std_onehot/result/trial1_train_augmented.csv', index=False)
print("Augmentation completed.")

