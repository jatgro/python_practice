"""
Network Intrusion Detection System
====================================
Predictive model to distinguish between attack (bad) and normal (good) network connections
using the KDD Cup 1999 dataset.

Process Steps:
    1. Data Pre-processing
    2. Data Correlation
    3. Feature Selection
    4. Modelling (NB, DT, RF, SVM, LR, GB)
    5. Validation & Comparison

Dataset: KDD Cup 1999 — https://www.kaggle.com/datasets/kavl31/kdd-cup-1999-data
Reference: https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/
"""

import gzip
import os
import shutil
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KDD Cup 1999 column names (the CSV ships without a header)
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
]

# Attack categories → binary mapping
ATTACK_TYPES = {
    "normal": "Normal",
    # DOS attacks
    "back": "Attack", "land": "Attack", "neptune": "Attack", "pod": "Attack",
    "smurf": "Attack", "teardrop": "Attack",
    # Probe attacks
    "ipsweep": "Attack", "nmap": "Attack", "portsweep": "Attack", "satan": "Attack",
    # R2L attacks
    "ftp_write": "Attack", "guess_passwd": "Attack", "imap": "Attack",
    "multihop": "Attack", "phf": "Attack", "spy": "Attack",
    "warezclient": "Attack", "warezmaster": "Attack",
    # U2R attacks
    "buffer_overflow": "Attack", "loadmodule": "Attack", "perl": "Attack",
    "rootkit": "Attack",
}


# ─── 0. Dataset Download ─────────────────────────────────────────────────────

# Original UCI ML Repository mirror (no API key required)
DATASET_URL = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"


def download_dataset(dest_dir: str | None = None) -> str:
    """Download the KDD Cup 1999 (10 %) dataset if it isn't already present.

    Returns the path to the uncompressed CSV file.
    """
    if dest_dir is None:
        dest_dir = os.path.dirname(__file__)

    csv_path = os.path.join(dest_dir, "kddcup.data_10_percent.csv")
    gz_path = csv_path + ".gz"

    if os.path.isfile(csv_path):
        print(f"[+] Dataset already exists → {csv_path}")
        return csv_path

    print(f"[↓] Downloading KDD Cup 1999 (10 %) from UCI …")
    print(f"    URL : {DATASET_URL}")
    print(f"    Dest: {gz_path}")
    urllib.request.urlretrieve(DATASET_URL, gz_path)
    print(f"[+] Download complete ({os.path.getsize(gz_path) / 1e6:.1f} MB compressed)")

    # Decompress .gz → .csv
    print("[↓] Decompressing …")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"[+] Ready → {csv_path}  ({os.path.getsize(csv_path) / 1e6:.1f} MB)")

    return csv_path


# ─── 1. Data Loading & Pre-processing ────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the KDD Cup 1999 dataset from a CSV file."""
    print("=" * 70)
    print("STEP 1: DATA PRE-PROCESSING")
    print("=" * 70)

    if filepath.endswith(".gz"):
        df = pd.read_csv(filepath, names=KDD_COLUMNS, header=None, compression="gzip")
    else:
        df = pd.read_csv(filepath, names=KDD_COLUMNS, header=None)

    print(f"\n[+] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame, sample_size: int = 100_000) -> pd.DataFrame:
    """Clean, encode, and prepare the dataset."""
    # Strip trailing '.' from labels (present in the raw KDD data)
    df["label"] = df["label"].str.strip().str.rstrip(".")

    # Map to binary classification: Normal vs Attack
    df["binary_label"] = df["label"].map(ATTACK_TYPES).fillna("Attack")
    print(f"[+] Class distribution:\n{df['binary_label'].value_counts()}\n")

    # Sample for tractability (full dataset is ~5 M rows)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[+] Sampled {sample_size:,} rows for tractable computation")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[+] Removed {before - len(df):,} duplicate rows  →  {len(df):,} remaining")

    # Encode categorical features
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ("label", "binary_label")]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    print(f"[+] Encoded {len(categorical_cols)} categorical features: {categorical_cols}")

    # Encode target
    df["target"] = LabelEncoder().fit_transform(df["binary_label"])  # 0=Attack, 1=Normal

    # Drop original text labels
    df.drop(columns=["label", "binary_label"], inplace=True)

    print(f"[+] Final pre-processed shape: {df.shape}")
    return df


# ─── 2. Data Correlation ─────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> None:
    """Compute and visualise the correlation matrix."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA CORRELATION")
    print("=" * 70)

    corr = df.drop(columns=["target"]).corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.1, fmt=".1f")
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Correlation heatmap saved → {path}")

    # Highly correlated pairs (|r| > 0.9)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [
        (col, upper.index[row], upper.iloc[row, col_idx])
        for col_idx, col in enumerate(upper.columns)
        for row in range(len(upper.index))
        if abs(upper.iloc[row, col_idx]) > 0.9
    ]
    if high_corr:
        print(f"[+] Found {len(high_corr)} highly correlated feature pairs (|r| > 0.9):")
        for c1, c2, r in high_corr[:10]:
            print(f"    {c1} ↔ {c2}  r={r:.3f}")


# ─── 3. Feature Selection ────────────────────────────────────────────────────

def select_features(
    X: pd.DataFrame, y: pd.Series, k: int = 20
) -> tuple[pd.DataFrame, list[str]]:
    """Select top-k features using chi-squared test."""
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE SELECTION")
    print("=" * 70)

    # Chi-squared requires non-negative values → shift if needed
    X_pos = X - X.min()

    selector = SelectKBest(chi2, k=k)
    selector.fit(X_pos, y)

    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    selected = scores.head(k).index.tolist()

    print(f"[+] Top {k} features (chi² scores):")
    for i, (feat, score) in enumerate(scores.head(k).items(), 1):
        print(f"    {i:>2}. {feat:<35s} {score:>14,.1f}")

    # Plot
    plt.figure(figsize=(12, 6))
    scores.head(k).plot(kind="barh", color="steelblue")
    plt.xlabel("Chi² Score")
    plt.title(f"Top {k} Features by Chi² Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Feature importance chart saved → {path}")

    return X[selected], selected


# ─── 4. Modelling ────────────────────────────────────────────────────────────

MODELS = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
}


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train each model and collect evaluation metrics."""
    print("\n" + "=" * 70)
    print("STEP 4: MODELLING & EVALUATION")
    print("=" * 70)

    results = []

    for name, model in MODELS.items():
        print(f"\n── {name} {'─' * (50 - len(name))}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Cross-validation (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "CV Mean": cv_scores.mean(),
                "CV Std": cv_scores.std(),
            }
        )

        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  F1-Score   : {f1:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Attack', 'Normal'])}")

    return pd.DataFrame(results)


# ─── 5. Validation & Comparison ──────────────────────────────────────────────

def compare_models(results_df: pd.DataFrame) -> None:
    """Visualise and compare model performance."""
    print("\n" + "=" * 70)
    print("STEP 5: VALIDATION & COMPARISON")
    print("=" * 70)

    print("\n" + results_df.to_string(index=False))

    best = results_df.loc[results_df["F1-Score"].idxmax()]
    print(f"\n★  Best model by F1-Score: {best['Model']}  (F1={best['F1-Score']:.4f})")

    # Bar chart comparison
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, metric in zip(axes, metrics):
        bars = ax.barh(results_df["Model"], results_df[metric], color="steelblue")
        ax.set_xlim(0, 1.05)
        ax.set_title(metric, fontsize=13)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
    fig.suptitle("Model Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparison chart saved → {path}")

    # Cross-validation box comparison
    plt.figure(figsize=(10, 5))
    x = range(len(results_df))
    plt.errorbar(
        x,
        results_df["CV Mean"],
        yerr=results_df["CV Std"],
        fmt="o",
        capsize=5,
        color="steelblue",
        markersize=8,
    )
    plt.xticks(list(x), results_df["Model"], rotation=30, ha="right")
    plt.ylabel("CV Accuracy")
    plt.title("5-Fold Cross-Validation Accuracy ± Std", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cv_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] CV comparison chart saved → {path}")

    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, _) in zip(axes.ravel(), MODELS.items()):
        ax.set_title(name)
    plt.tight_layout()
    plt.close()

    # Save results to CSV
    path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(path, index=False)
    print(f"[+] Results table saved → {path}")


def plot_confusion_matrices(
    X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Plot confusion matrix for each trained model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, model) in zip(axes.ravel(), MODELS.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Attack", "Normal"], yticklabels=["Attack", "Normal"],
        )
        ax.set_title(name, fontsize=12)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices", fontsize=16)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrices saved → {path}")


def plot_roc_curves(
    X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Plot ROC curves for all models on a single figure."""
    plt.figure(figsize=(10, 7))
    for name, model in MODELS.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models", fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] ROC curves saved → {path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main(data_path: str) -> None:
    """Execute the full intrusion detection pipeline."""

    # 1. Load & pre-process
    df = load_data(data_path)
    df = preprocess(df)

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # 2. Correlation analysis
    correlation_analysis(df)

    # 3. Feature selection
    X_selected, selected_features = select_features(X, y, k=20)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_selected), columns=selected_features
    )

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[+] Train set: {X_train.shape[0]:,}  |  Test set: {X_test.shape[0]:,}")

    # 4. Train & evaluate models
    results_df = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5. Compare models
    compare_models(results_df)
    plot_confusion_matrices(X_test, y_test)
    plot_roc_curves(X_test, y_test)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — all outputs in:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Network Intrusion Detection using KDD Cup 1999 dataset"
    )
    parser.add_argument(
        "data",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "kddcup.data_10_percent.csv"),
        help="Path to the KDD Cup 1999 CSV file (default: kddcup.data_10_percent.csv in same dir)",
    )
    args = parser.parse_args()

    # Auto-download if the file doesn't exist
    if not os.path.isfile(args.data):
        print(f"Dataset not found at '{args.data}' — downloading automatically …\n")
        args.data = download_dataset()

    main(args.data)
