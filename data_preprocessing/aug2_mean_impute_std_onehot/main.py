import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy import sparse

# === USER PARAMETERS ===
POP_AUG_RATIO = 2       # 인기 뉴스 증강 배율 (1 = 유지, 2 = 2배)
NONPOP_AUG_RATIO = 1.5  # 비인기 뉴스 증강 배율 (1 = 유지, 1.5 = 1.5배)

# File paths
INPUT_CSV = '/home/kanghosung/hw1_patt/pattern-recognition/data/train.csv'
OUTPUT_DIR = os.path.join(os.getcwd(), 'result', 'trial2')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEST_CSV = os.path.join(OUTPUT_DIR, f'{POP_AUG_RATIO}_{NONPOP_AUG_RATIO}_trial2_test.csv')
TRAIN_CSV = os.path.join(OUTPUT_DIR, f'{POP_AUG_RATIO}_{NONPOP_AUG_RATIO}_trial2_train.csv')

# 1. Load & label
df_raw = pd.read_csv(INPUT_CSV)
df_raw['target'] = (df_raw['shares'] > 1400).astype(int)
df_raw = df_raw.drop(columns=['id','shares','y'], errors='ignore')

# 2. Define preprocess pipelines
categorical_cols = ['data_channel','weekday']
numerical_cols   = [c for c in df_raw.columns if c not in categorical_cols + ['target']]

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=True))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numerical_cols),
    ('cat', cat_pipe, categorical_cols)
])

# 3. Preprocess all data
X_all = df_raw.drop(columns=['target'])
y_all = df_raw['target'].values
X_proc_all = preprocessor.fit_transform(X_all)

# build DataFrame of processed features
num_feats = numerical_cols
cat_feats = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_feats = np.concatenate([num_feats, cat_feats])
if sparse.issparse(X_proc_all):
    df_proc = pd.DataFrame.sparse.from_spmatrix(X_proc_all, columns=all_feats)
else:
    df_proc = pd.DataFrame(X_proc_all, columns=all_feats)
df_proc['target'] = y_all

# 4. Train/Test split
df_train, df_test = train_test_split(
    df_proc, test_size=0.2, random_state=42, stratify=df_proc['target']
)
df_test.to_csv(TEST_CSV, index=False)

# 5. Reverse one-hot to categories for train grouping
orig = df_train.copy()
# extract category values per row
for cat in categorical_cols:
    cols = [c for c in all_feats if c.startswith(f"{cat}_")]
    # array of one-hot submatrix
    arr = orig[cols].values
    # find index of 1 per row; map back to category name
    values = [cols[row.argmax()].split('_',1)[1] for row in arr]
    orig[cat] = values

# 6. Compute synth counts for each class
n_pos = (orig.target==1).sum()
n_neg = (orig.target==0).sum()
n_synth_pos = int(n_pos*(POP_AUG_RATIO-1))
n_synth_neg = int(n_neg*(NONPOP_AUG_RATIO-1))

# 7. Group-based synthesis for pos/neg
synth_list = []
for cls, n_synth in [(1, n_synth_pos), (0, n_synth_neg)]:
    if n_synth <= 0:
        continue
    grp = orig[orig.target==cls].groupby(categorical_cols)
    counts = grp.size()
    synth_counts = (counts/counts.sum()*n_synth).round().astype(int)
    for (ch, wd), cnt in synth_counts.items():
        if cnt<=0:
            continue
        data_grp = grp.get_group((ch,wd))[numerical_cols]
        mu = data_grp.mean().values
        cov = np.cov(data_grp.values, rowvar=False)
        eps = 1e-6*np.trace(cov)/len(cov)
        cov += np.eye(len(numerical_cols))*eps
        try:
            samples = np.random.multivariate_normal(mu, cov, size=cnt)
        except np.linalg.LinAlgError:
            var = np.diag(cov)
            samples = np.random.normal(mu, np.sqrt(var), size=(cnt,len(mu)))
        df_s = pd.DataFrame(samples, columns=numerical_cols)
        df_s['data_channel'] = ch
        df_s['weekday'] = wd
        df_s['target'] = cls
        synth_list.append(df_s)

df_synth = pd.concat(synth_list, ignore_index=True)

# 8. Text-feature jitter on positive only
text_cols = [c for c in numerical_cols if any(k in c for k in ['token','subjectivity','polarity'])]
mask = df_synth['target']==1
for col in text_cols:
    scale = 0.01*(df_synth[col].max()-df_synth[col].min())
    df_synth.loc[mask, col] += np.random.normal(scale=scale, size=mask.sum())

# 9. Combine and finalize train
df_train_aug = pd.concat([orig.drop(columns=categorical_cols), df_synth], ignore_index=True)
# drop reversed cat cols if present, ensure processed features only
df_train_aug = df_train_aug[all_feats.tolist()+['target']]

df_train_aug.to_csv(TRAIN_CSV, index=False)
print(f"Saved test to {TEST_CSV}")
print(f"Saved augmented train to {TRAIN_CSV}")
