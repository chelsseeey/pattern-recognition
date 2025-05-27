import warnings, os, sys
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

# ───────────────── CONFIG ───────────────── #
DATA_PATH          = "/home/kanghosung/hw1_patt/data_preprocessing/result/trial1_train.csv"
TEST_SIZE          = 0.20          # train-test split 비율
RANDOM_STATE       = 42            # 재현용 시드
TOP_K_FEATURES     = 20            # feature importance 시각화 개수

# GBM 하이퍼파라미터 그리드
GBM_PARAM_GRID = {
    "clf__n_estimators"     : [200, 400],
    "clf__learning_rate"    : [0.01, 0.05, 0.1],
    "clf__max_depth"        : [3, 5],
    "clf__subsample"        : [0.8, 1.0],
    "clf__min_samples_split": [2, 4],
    "clf__min_samples_leaf" : [1, 2],
    "clf__max_features"     : ["sqrt"]
}

# 출력 폴더 기본 경로 (timestamp는 자동 부여)
OUTPUT_DIR_BASE    = "/home/kanghosung/hw1_patt/GradientBoostingClassifier/log"
# ─────────────────────────────────────────── #


# ──────────────────────── 경로 / 로그 설정 ──────────────────────── #
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = os.path.join(OUTPUT_DIR_BASE, f"run_{timestamp}")
os.makedirs(output_root, exist_ok=True)

log_path = os.path.join(output_root, "run.log")
conf_path = os.path.join(output_root, "confusion_matrix.png")
roc_path  = os.path.join(output_root, "roc_curve.png")
pr_path   = os.path.join(output_root, "pr_curve.png")
fi_path   = os.path.join(output_root, "feature_importance.png")

# ──────────────────────── 데이터 로드 ──────────────────────── #
df  = pd.read_csv(DATA_PATH)
y   = df["target"]
X   = df.drop(columns=["target"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# ──────────────────────── 전처리 파이프라인 ──────────────────────── #
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

# ──────────────────────── 전체 파이프라인 ──────────────────────── #
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
])

# ──────────────────────── 데이터 분할 ──────────────────────── #
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ──────────────────────── GridSearchCV ──────────────────────── #
with open(log_path, "w") as log:
    print("▶ GridSearchCV 시작…", file=log)
    grid = GridSearchCV(pipe, GBM_PARAM_GRID, cv=3, scoring="roc_auc",
                        n_jobs=-1, verbose=1)
    grid.fit(X_trainval, y_trainval)

    print("Best params:", grid.best_params_, "\n", file=log)
    best_pipe = grid.best_estimator_

    # ───────── 5-fold CV 평가 ───────── #
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_l, f1_l, auc_l = [], [], []

    print("5-Fold CV:", file=log)
    for i, (tr, vl) in enumerate(tqdm(cv.split(X_trainval, y_trainval), total=5)):
        best_pipe.fit(X_trainval.iloc[tr], y_trainval.iloc[tr])
        y_pred = best_pipe.predict(X_trainval.iloc[vl])
        y_prob = best_pipe.predict_proba(X_trainval.iloc[vl])[:, 1]

        acc_l.append(accuracy_score(y_trainval.iloc[vl], y_pred))
        f1_l .append(f1_score   (y_trainval.iloc[vl], y_pred))
        auc_l.append(roc_auc_score(y_trainval.iloc[vl], y_prob))

        print(f"[Fold {i+1}] ACC={acc_l[-1]:.4f} | F1={f1_l[-1]:.4f} | AUC={auc_l[-1]:.4f}",
              file=log)

    comp = np.mean([acc_l, f1_l, auc_l], axis=0)
    print(f"Composite mean: {comp.mean():.4f}\n", file=log)

    # ───────── Hold-out 테스트 ───────── #
    best_pipe.fit(X_trainval, y_trainval)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    comp_test = (acc + f1 + auc) / 3

    print("Hold-out Test:", file=log)
    for name, val in zip(["ACC", "F1", "AUC", "Composite"],
                         [acc, f1, auc, comp_test]):
        print(f"{name}: {val:.4f}", file=log)

    # ───────── Classification report ───────── #
    print("\nClassification report\n", classification_report(y_test, y_pred),
          file=log)

# ──────────────────────── 그래프/지표 저장 ──────────────────────── #
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0","1"], yticklabels=["0","1"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(conf_path); plt.close()

# Feature importances (상위 TOP_K_FEATURES)
feat_names = best_pipe.named_steps["pre"].get_feature_names_out()
importances = best_pipe.named_steps["clf"].feature_importances_
top = np.argsort(importances)[::-1][:TOP_K_FEATURES]
plt.figure(figsize=(8,6))
sns.barplot(x=importances[top], y=np.array(feat_names)[top])
plt.title(f"Top-{TOP_K_FEATURES} Feature Importances")
plt.tight_layout(); plt.savefig(fi_path); plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(); plt.plot(fpr, tpr)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.tight_layout(); plt.savefig(roc_path); plt.close()

# PR curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure(); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.tight_layout(); plt.savefig(pr_path); plt.close()

print(f"done")
