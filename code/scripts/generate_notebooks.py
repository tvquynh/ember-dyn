"""Generate four model quickstart notebooks for EMBER-Dyn.

Notebooks (each runnable on Colab free CPU tier in <10 min):
  01_quickstart_lightgbm.ipynb       - T1 LightGBM static, target AUC 0.987
  02_quickstart_xgboost.ipynb        - T1 XGBoost static, target AUC 0.985
  03_quickstart_mlp_pytorch.ipynb    - T1 MLP-3 (PyTorch), target AUC 0.972
  04_quickstart_rf_sklearn.ipynb     - T1 Random Forest, target AUC 0.978

The dataset overview and T5 protocol-verification notebooks are maintained
separately and are not produced by this generator. Outputs are written to the
repository-root notebooks/ directory, where the notebooks ship.
"""

import json
from pathlib import Path

ROOT = Path(".")
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

# Common boilerplate for all notebooks
COMMON_HEADER = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# {title}\n",
            "\n",
            "**Dataset**: EMBER-Dyn ({task})  \n",
            "**Target metric**: {target_metric}  \n",
            "**Runtime**: ~5-10 min on Colab free CPU tier\n",
            "\n",
            "This notebook loads a local EMBER-Dyn candidate release, trains a {model} ",
            "baseline on the 2,568-dimensional static features, and reproduces ",
            "the paper Table 4 result.\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Install dependencies"],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "!pip install -q pandas pyarrow scikit-learn {extra_deps}\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Load EMBER-Dyn dataset"],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import pyarrow.parquet as pq\n",
            "import numpy as np\n",
            "from sklearn.metrics import roc_auc_score, f1_score\n",
            "\n",
            "# Obtain the author-approved archive and verify its checksum.\n",
            "# The Zenodo archival release is forthcoming; verify its DOI and checksum before use.\n",
            "# Then point DATA_DIR at the extracted dataset/static folder.\n",
            "DATA_DIR = 'dataset/static'  # adjust to your local extraction path\n",
            "df_train = pq.read_table(f'{DATA_DIR}/train.parquet').to_pandas()\n",
            "df_val   = pq.read_table(f'{DATA_DIR}/val.parquet').to_pandas()\n",
            "df_test  = pq.read_table(f'{DATA_DIR}/test.parquet').to_pandas()\n",
            "df_chal  = pq.read_table(f'{DATA_DIR}/challenge.parquet').to_pandas()\n",
            "\n",
            "print(f'Train: {len(df_train):,} samples')\n",
            "print(f'Val:   {len(df_val):,} samples')\n",
            "print(f'Test:  {len(df_test):,} samples')\n",
            "print(f'Chal:  {len(df_chal):,} samples')",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Extract features and labels"],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "feat_cols = [c for c in df_train.columns if c.startswith('feature_')]\n",
            "print(f'Feature dimensions: {len(feat_cols)}')  # 2568\n",
            "\n",
            "def split_X_y(df):\n",
            "    X = df[feat_cols].values.astype('float32')\n",
            "    y = df['label'].values.astype('int32')\n",
            "    return X, y\n",
            "\n",
            "X_train, y_train = split_X_y(df_train)\n",
            "X_val,   y_val   = split_X_y(df_val)\n",
            "X_test,  y_test  = split_X_y(df_test)\n",
            "X_chal,  y_chal  = split_X_y(df_chal)\n",
            "\n",
            "print(f'X_train shape: {X_train.shape}, y_train balance: {y_train.mean():.3f}')",
        ],
    },
]

def safe_format(line, **kwargs):
    """Format `line` with kwargs only if it contains a placeholder."""
    if "{" not in line:
        return line
    try:
        return line.format(**kwargs)
    except (KeyError, IndexError):
        return line  # placeholder unrelated to our kwargs

def make_notebook(title, task, model, target_metric, extra_deps, train_cells):
    """Build one notebook by combining common header + model-specific cells."""
    header = []
    for cell in COMMON_HEADER:
        c = json.loads(json.dumps(cell))  # deep copy
        c["source"] = [
            safe_format(
                line,
                title=title, task=task, model=model,
                target_metric=target_metric, extra_deps=extra_deps,
            )
            for line in c["source"]
        ]
        header.append(c)

    nb = {
        "cells": header + train_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb

# -------------------------------------------------------------------
# Notebook 01: LightGBM
# -------------------------------------------------------------------
nb01 = make_notebook(
    title="EMBER-Dyn Quickstart 01: LightGBM",
    task="T1 Binary detection",
    model="LightGBM",
    target_metric="AUC ≈ 0.987",
    extra_deps="lightgbm",
    train_cells=[
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 4. Train LightGBM with 200 boosting rounds + early stopping"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "import lightgbm as lgb\n",
            "\n",
            "model = lgb.LGBMClassifier(\n",
            "    n_estimators=200,\n",
            "    learning_rate=0.05,\n",
            "    num_leaves=63,\n",
            "    random_state=42,\n",
            ")\n",
            "model.fit(\n",
            "    X_train, y_train,\n",
            "    eval_set=[(X_val, y_val)],\n",
            "    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(50)],\n",
            ")",
        ]},
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 5. Evaluate on test + challenge splits"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "for split_name, X, y in [('test', X_test, y_test), ('challenge', X_chal, y_chal)]:\n",
            "    p = model.predict_proba(X)[:, 1]\n",
            "    auc = roc_auc_score(y, p)\n",
            "    f1 = f1_score(y, p > 0.5)\n",
            "    print(f'{split_name:>10s}: AUC={auc:.4f}  F1={f1:.4f}')",
        ]},
        {"cell_type": "markdown", "metadata": {},
         "source": [
            "## Expected output\n",
            "\n",
            "```\n",
            "       test: AUC=0.987  F1=0.964\n",
            "  challenge: AUC=0.943  F1=0.918\n",
            "```\n",
            "\n",
            "Matches paper Table 4 LightGBM (static) row.",
         ]},
    ],
)
(NB_DIR / "01_quickstart_lightgbm.ipynb").write_text(json.dumps(nb01, indent=1))

# -------------------------------------------------------------------
# Notebook 02: XGBoost
# -------------------------------------------------------------------
nb02 = make_notebook(
    title="EMBER-Dyn Quickstart 02: XGBoost",
    task="T1 Binary detection",
    model="XGBoost",
    target_metric="AUC ≈ 0.985",
    extra_deps="xgboost",
    train_cells=[
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 4. Train XGBoost (200 rounds, early stopping)"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "import xgboost as xgb\n",
            "\n",
            "model = xgb.XGBClassifier(\n",
            "    n_estimators=200,\n",
            "    learning_rate=0.05,\n",
            "    max_depth=6,\n",
            "    tree_method='hist',\n",
            "    early_stopping_rounds=10,\n",
            "    random_state=42,\n",
            ")\n",
            "model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)",
        ]},
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 5. Evaluate"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "for split_name, X, y in [('test', X_test, y_test), ('challenge', X_chal, y_chal)]:\n",
            "    p = model.predict_proba(X)[:, 1]\n",
            "    auc = roc_auc_score(y, p)\n",
            "    f1 = f1_score(y, p > 0.5)\n",
            "    print(f'{split_name:>10s}: AUC={auc:.4f}  F1={f1:.4f}')",
        ]},
    ],
)
(NB_DIR / "02_quickstart_xgboost.ipynb").write_text(json.dumps(nb02, indent=1))

# -------------------------------------------------------------------
# Notebook 03: MLP-3 (PyTorch)
# -------------------------------------------------------------------
nb03 = make_notebook(
    title="EMBER-Dyn Quickstart 03: MLP-3 (PyTorch)",
    task="T1 Binary detection",
    model="3-layer MLP",
    target_metric="AUC ≈ 0.972",
    extra_deps="torch",
    train_cells=[
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 4. Define 3-layer MLP"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "from torch.utils.data import DataLoader, TensorDataset\n",
            "\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print(f'Device: {device}')\n",
            "\n",
            "class MLP(nn.Module):\n",
            "    def __init__(self, dim_in=2568, dim_h=512, dim_out=2):\n",
            "        super().__init__()\n",
            "        self.net = nn.Sequential(\n",
            "            nn.Linear(dim_in, dim_h), nn.ReLU(), nn.Dropout(0.3),\n",
            "            nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Dropout(0.3),\n",
            "            nn.Linear(dim_h, dim_out),\n",
            "        )\n",
            "    def forward(self, x): return self.net(x)\n",
            "\n",
            "model = MLP().to(device)",
        ]},
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 5. Train (20 epochs, batch 256)"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "X_train_t = torch.from_numpy(X_train).to(device)\n",
            "y_train_t = torch.from_numpy(y_train).long().to(device)\n",
            "loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)\n",
            "\n",
            "opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
            "loss_fn = nn.CrossEntropyLoss()\n",
            "\n",
            "for epoch in range(20):\n",
            "    model.train()\n",
            "    total_loss = 0.0\n",
            "    for xb, yb in loader:\n",
            "        opt.zero_grad()\n",
            "        loss = loss_fn(model(xb), yb)\n",
            "        loss.backward()\n",
            "        opt.step()\n",
            "        total_loss += loss.item()\n",
            "    if epoch % 5 == 0 or epoch == 19:\n",
            "        print(f'epoch {epoch}: loss={total_loss / len(loader):.4f}')",
        ]},
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 6. Evaluate"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "model.eval()\n",
            "with torch.no_grad():\n",
            "    for split_name, X, y in [('test', X_test, y_test), ('challenge', X_chal, y_chal)]:\n",
            "        X_t = torch.from_numpy(X).to(device)\n",
            "        p = torch.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()\n",
            "        auc = roc_auc_score(y, p)\n",
            "        f1 = f1_score(y, p > 0.5)\n",
            "        print(f'{split_name:>10s}: AUC={auc:.4f}  F1={f1:.4f}')",
        ]},
    ],
)
(NB_DIR / "03_quickstart_mlp_pytorch.ipynb").write_text(json.dumps(nb03, indent=1))

# -------------------------------------------------------------------
# Notebook 04: Random Forest (sklearn)
# -------------------------------------------------------------------
nb04 = make_notebook(
    title="EMBER-Dyn Quickstart 04: Random Forest",
    task="T1 Binary detection",
    model="Random Forest",
    target_metric="AUC ≈ 0.978",
    extra_deps="",
    train_cells=[
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 4. Train Random Forest (200 trees)"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "from sklearn.ensemble import RandomForestClassifier\n",
            "\n",
            "model = RandomForestClassifier(\n",
            "    n_estimators=200, max_depth=None, n_jobs=-1, random_state=42,\n",
            ")\n",
            "model.fit(X_train, y_train)",
        ]},
        {"cell_type": "markdown", "metadata": {},
         "source": ["## 5. Evaluate"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "for split_name, X, y in [('test', X_test, y_test), ('challenge', X_chal, y_chal)]:\n",
            "    p = model.predict_proba(X)[:, 1]\n",
            "    auc = roc_auc_score(y, p)\n",
            "    f1 = f1_score(y, p > 0.5)\n",
            "    print(f'{split_name:>10s}: AUC={auc:.4f}  F1={f1:.4f}')",
        ]},
    ],
)
(NB_DIR / "04_quickstart_rf_sklearn.ipynb").write_text(json.dumps(nb04, indent=1))

print(f"Generated model quickstart notebooks in {NB_DIR}:")
for nb in sorted(NB_DIR.glob("*.ipynb")):
    print(f"  {nb.name}  ({nb.stat().st_size / 1024:.1f} KB)")
