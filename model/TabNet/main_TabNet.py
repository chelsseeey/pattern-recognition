import os
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Argument parsing for GPU selection
parser = argparse.ArgumentParser()
parser.add_argument('--g', type=str, default='0', help='GPU index (0-7)')
args = parser.parse_args()

# Set visible CUDA devices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

# Set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Output path setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = "/home/kanghosung/hw1_patt/TabNet/log"
os.makedirs(output_root, exist_ok=True)

log_path = os.path.join(output_root, f"log_{timestamp}.txt")
conf_matrix_path = os.path.join(output_root, f"confusion_matrix_{timestamp}.png")

# Load dataset
train_src = '/home/kanghosung/hw1_patt/data/train.csv'
df = pd.read_csv(train_src)

X = df.drop(['id', 'shares', 'y'], axis=1)
y = df['y'].values

# Column split
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = ['data_channel', 'weekday']

# Preprocessing
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(imputer.fit_transform(X[num_cols]))

cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    cat_encoders[col] = le

# Train/validation/test split: 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(X.values, y, test_size=0.4, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# TabNet Model
clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10,
    seed=42,
    device_name=DEVICE
)

# Training
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=["valid"],
    eval_metric=["auc"],
    max_epochs=200,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0
)

# Prediction & Evaluation on test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Save results
with open(log_path, "w") as log_file:
    for name, val in zip(["Accuracy", "F1 Score", "ROC AUC"], [acc, f1, auc]):
        line = f"{name} : {val:.4f}"
        print(line)
        log_file.write(line + "\n")

    report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
    print("\nClassification Report:\n", report)
    log_file.write("\n" + report)

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

print("done")
