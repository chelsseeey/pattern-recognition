import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Timestamp and output path setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = "/home/kanghosung/hw1_patt/GBM/log/gridSearch/trial1_tain_augmented"
os.makedirs(output_root, exist_ok=True)

log_path = os.path.join(output_root, f"log_{timestamp}.txt")
conf_matrix_path = os.path.join(output_root, f"confusion_matrix_{timestamp}.png")
roc_path = os.path.join(output_root, f"roc_curve_{timestamp}.png")
pr_path = os.path.join(output_root, f"pr_curve_{timestamp}.png")
fi_path = os.path.join(output_root, f"feature_importance_{timestamp}.png")

#train_src = '/home/kanghosung/hw1_patt/data_preprocessing/result/trial1_train.csv'
train_src = '/home/kanghosung/hw1_patt/pattern-recognition/data_preprocessing/aug_mean_impute_std_onehot/result/trial1_train_augmented.csv'
df = pd.read_csv(train_src)

X = df.drop(columns=['target'])
y = df['target']

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


num_cols = X.columns.tolist()

numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols)
])

pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=42))
])

# GridSearchCV setup
param_grid = {
    'clf__n_estimators': [200, 400],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5],
    'clf__subsample': [0.8, 1.0],
    'clf__min_samples_split': [2, 4],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt']
}

print("Starting GridSearchCV...")
grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search.fit(X_trainval, y_trainval)

best_pipe = grid_search.best_estimator_

with open(log_path, "w") as log_file:
    log_file.write("Best Parameters from GridSearchCV:\n")
    log_file.write(str(grid_search.best_params_) + "\n\n")
    print("Best Parameters:", grid_search.best_params_)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc, f1, auc = [], [], []

    log_file.write("5-Fold CV Evaluation:\n")
    print("\n5-Fold CV Evaluation:")
    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_trainval, y_trainval), total=5)):
        X_tr, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_tr, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        best_pipe.fit(X_tr, y_tr)
        y_pred = best_pipe.predict(X_val)
        y_prob = best_pipe.predict_proba(X_val)[:, 1]

        acc.append(accuracy_score(y_val, y_pred))
        f1.append(f1_score(y_val, y_pred))
        auc.append(roc_auc_score(y_val, y_prob))

    comp = (np.array(acc) + np.array(f1) + np.array(auc)) / 3

    for i in range(5):
        line = f"[Fold {i+1}] Accuracy: {acc[i]:.4f}, F1: {f1[i]:.4f}, AUC: {auc[i]:.4f}, Composite: {comp[i]:.4f}"
        print(line)
        log_file.write(line + "\n")
    avg_line = f"\nAverage Composite Score: {comp.mean():.4f}"
    print(avg_line)
    log_file.write(avg_line + "\n")


    best_pipe.fit(X_trainval, y_trainval)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    comp = (acc + f1 + auc) / 3

    print("\nHoldout Test Evaluation")
    log_file.write("\nHoldout Test Evaluation\n")
    for metric_name, value in zip(["Accuracy", "F1 Score", "ROC AUC", "Composite"], [acc, f1, auc, comp]):
        line = f"{metric_name} : {value:.4f}"
        print(line)
        log_file.write(line + "\n")

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
    print("\nClassification Report:\n", report)
    log_file.write("\nClassification Report:\n" + report + "\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # Feature Importance
    feature_names = best_pipe.named_steps["pre"].get_feature_names_out()
    importances = best_pipe.named_steps["clf"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[sorted_idx][:20], y=np.array(feature_names)[sorted_idx][:20])
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(rec, prec, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

print("All tasks completed.")
