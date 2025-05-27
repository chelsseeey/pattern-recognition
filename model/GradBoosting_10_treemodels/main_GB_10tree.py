import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# suppress xgboost deprecation warnings
warnings.filterwarnings("ignore", message="`use_label_encoder` is deprecated")


from sklearn.utils._tags import _safe_tags
from xgboost import XGBClassifier
XGBClassifier.__sklearn_tags__ = lambda self: _safe_tags(self)


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    StackingClassifier, GradientBoostingClassifier,
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Paths and output setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = "/home/kanghosung/hw1_patt/pattern-recognition/model/GradBoosting_10_treemodels/result/"
os.makedirs(output_root, exist_ok=True)
log_path = os.path.join(output_root, f"log_{timestamp}.txt")
conf_matrix_path = os.path.join(output_root, f"confusion_matrix_{timestamp}.png")
roc_path = os.path.join(output_root, f"roc_curve_{timestamp}.png")
pr_path = os.path.join(output_root, f"pr_curve_{timestamp}.png")

# 2. Load data
train_src = '/home/kanghosung/hw1_patt/data_preprocessing/result/trial1_train.csv'
df = pd.read_csv(train_src)
X = df.drop(columns=['target'])
y = df['target']

# 3. Split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Identify features
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

# 5. Preprocessor
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

# 6. Define 10 SOTA base models
estimators = [
  #  ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('lgbm', RandomForestClassifier(n_estimators=100, random_state=42)),    # replace with LGBM if available
    ('cat', ExtraTreesClassifier(n_estimators=100, random_state=42)),       # replace with CatBoost if available
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=200, random_state=42)),
    ('hgb', HistGradientBoostingClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('mlp', MLPClassifier(max_iter=500, random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42)),
    ('bag', BaggingClassifier(random_state=42))
]

# 7. Meta-learner
meta_learner = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.8, min_samples_split=4, min_samples_leaf=2,
    max_features='sqrt', random_state=42
)

# 8. StackingClassifier
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1,
    passthrough=False
)

# 9. Full pipeline
pipe = Pipeline([
    ('pre', preprocessor),
    ('stack', stack)
])

# 10. Cross-validation and evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, f1s, aucs = [], [], []

with open(log_path, "w") as log_file:
    log_file.write("5-Fold Stacking CV:\n")
    for i, (tr_idx, val_idx) in enumerate(tqdm(cv.split(X_trainval, y_trainval), total=5), 1):
        X_tr, X_val = X_trainval.iloc[tr_idx], X_trainval.iloc[val_idx]
        y_tr, y_val = y_trainval.iloc[tr_idx], y_trainval.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:,1]
        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds))
        aucs.append(roc_auc_score(y_val, probs))
        log_file.write(f"[Fold {i}] Acc: {accs[-1]:.4f}, F1: {f1s[-1]:.4f}, AUC: {aucs[-1]:.4f}\n")
    comps = np.mean([accs, f1s, aucs], axis=0)
    log_file.write(f"Avg Composite: {comps.mean():.4f}\n")

    # Holdout evaluation
    pipe.fit(X_trainval, y_trainval)
    preds_test = pipe.predict(X_test)
    probs_test = pipe.predict_proba(X_test)[:,1]
    acc_h = accuracy_score(y_test, preds_test)
    f1_h = f1_score(y_test, preds_test)
    auc_h = roc_auc_score(y_test, probs_test)
    comp_h = np.mean([acc_h, f1_h, auc_h])
    log_file.write(f"Holdout - Acc: {acc_h:.4f}, F1: {f1_h:.4f}, AUC: {auc_h:.4f}, Comp: {comp_h:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds_test)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["0","1"], yticklabels=["0","1"])
    plt.title("Confusion Matrix")
    plt.savefig(conf_matrix_path); plt.close()

    # Classification Report
    rpt = classification_report(y_test, preds_test)
    log_file.write("\nClassification Report:\n" + rpt + "\n")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    plt.figure(); plt.plot(fpr, tpr, label="ROC Curve")
    plt.savefig(roc_path); plt.close()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, probs_test)
    plt.figure(); plt.plot(rec, prec, label="PR Curve")
    plt.savefig(pr_path); plt.close()

print("All tasks completed.")
