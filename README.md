<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PySpark-4.1.2-orange?logo=apache-spark" alt="PySpark">
  <img src="https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/RL-BCQ-purple" alt="BCQ">
  <img src="https://img.shields.io/badge/data-MovieLens%2025M-green" alt="MovieLens">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

<h1 align="center">SparRL</h1>
<h3 align="center">Spark + Offline Reinforcement Learning for Recommendation</h3>

<p align="center">
  <b>把推荐建模成序列决策问题，用 Spark 分布式构建轨迹，用 BCQ 离线强化学习策略。</b><br>
  <i>Recommendation as Sequential Decision Making — Spark for Trajectories, BCQ for Policy.</i>
</p>

---

## 🎯 Why SparRL?

| 传统推荐 | SparRL |
|----------|--------|
| (user, movie) → rating 监督学习 | (state, action) → reward 序列决策 |
| 拟合历史行为 | 学习最优推荐策略 |
| 单机 Pandas / SQL | **Spark 分布式窗口函数** |
| 离线评估只有 train/test split | **Online Simulation + 时间窗口学习曲线** |
| ItemCF / Two-Tower | **BCQ (VAE + Perturbation + Double Q)** |

**差异化**：Spark 工程 + 强化学习算法 + 推荐系统领域，三交叉。市面上几乎没有同类项目。

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SPARK LAYER (Distributed)                      │
│                                                                       │
│   MovieLens 25M  ───→  preprocess.py  ───→  trajectories.py  ───→ ope.py│
│   (25M ratings)       Feature Engineering    Window Functions ★     OPE │
│                       user/movie features    PARTITION BY userId      IPS │
│                       genre + year + stats   ROWS BETWEEN K          DM  │
│                       → parquet              → trajectory parquet    DR  │
│                                                                       │
│   ★ Key: Lazy Eval · Wide/Narrow Dependency · Shuffle · Broadcast Join   │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ trajectories & features
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PyTorch LAYER (BCQ Deep RL)                      │
│                                                                       │
│   ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐ │
│   │ State Encoder (GRU)│  │ BCQ Generator (VAE)│  │ Q-Network (×2)   │ │
│   │ encode rating      │──▶│ constrain policy    │──▶│ evaluate (s,a)  │ │
│   │ history → vector   │   │ via data support    │   │ → best action   │ │
│   └──────────────────┘   └──────────────────┘   └─────────────────┘ │
│                                                                       │
│   π(s) = argmax_{a ~ G(s)} Q(s, a + ξ(s,a))                          │
│                      ↑ VAE sampling      ↑ perturbation (|ξ| ≤ Φ)    │
│                                                                       │
│   ★ Key: Offline RL · Distribution Shift · Double DQN · VAE · Polyak │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
SparkPractice/
├── config.py                  # Global config (paths, hyperparams, AutoDL detection)
├── run_pipeline.py            # End-to-end pipeline (local/full/spark-only/train-only)
├── requirements.txt           # pip dependencies
├── itemcf_pandas.py           # ItemCF baseline (pure pandas, 0 ML libs)
│
├── spark/                     # Spark data pipeline
│   ├── preprocess.py          # Feature engineering (6 modules) + time-window split ★
│   ├── trajectories.py        # RL trajectory construction via window functions ★ Core
│   └── ope.py                 # Offline Policy Evaluation (IPS / DM / DR)
│
├── model/                     # PyTorch BCQ model
│   ├── embeddings.py          # Movie embedding + GRU State Encoder
│   ├── bcq.py                 # BCQ: VAE Generator + Perturbation + Double Q-Network
│   └── train.py               # Training loop (VAE pretrain → joint training)
│
├── eval/
│   └── evaluate.py            # NDCG / Recall / MRR + baselines + learning curve
│
├── data/ml-25m/ml-25m/        # MovieLens 25M (download required)
└── output/                    # Auto-generated intermediate files
    ├── movie_features.parquet
    ├── user_features.parquet
    ├── trajectories.parquet
    ├── bcq_model.pt
    └── online_windows/        # Time-window data for online simulation
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+ · **Java** 8/11/21 (PySpark requires JVM)
- **MovieLens 25M**: [download](https://files.grouplens.org/datasets/movielens/ml-25m.zip) (~250MB) → extract to `data/ml-25m/ml-25m/`

```bash
pip install -r requirements.txt
```

### Local Verification (5K users, ~10 min)

```bash
python run_pipeline.py --mode local --sample 5000
```

### Full Training (AutoDL / Cloud GPU)

```bash
python run_pipeline.py --mode full
```

### Step-by-Step

```bash
python spark/preprocess.py       # Spark feature engineering + time-window split
python spark/trajectories.py     # Trajectory construction (standard)
python model/train.py            # BCQ training (standard)
python eval/evaluate.py          # Evaluation with baselines
python spark/ope.py              # Offline Policy Evaluation
```

---

## 🕐 Online Simulation (Featured)

**用时间窗口模拟持续在线学习** — 逐轮评估模型在 "未来数据" 上的表现，画学习曲线。

```
Global Timeline [Jan]───[Apr]───[Jul]───[Oct]───[Jan]
                   W0      W1      W2      W3
                   │       │       │       │
Round 1: Train=W0 ─┘ Test=W1  (min data)
Round 2: Train=W0+W1 ────┘ Test=W2  (growing)
Round 3: Train=W0+W1+W2 ────┘ Test=W3  (full data)
                                        ↓
                      Learning Curve: NDCG↑ or ↓?
                      Data saturated or distribution shift?
```

```bash
# Generate time-window data
python run_pipeline.py --mode spark-only

# Build trajectories for each round
python spark/trajectories.py --online

# Train incrementally (each round warm-starts from previous)
python model/train.py --online

# Evaluate learning curve (ASCII plot)
python eval/evaluate.py --online

# Expected output:
#   轮1 (  3,200,000条) │████████░░░░░░░░░░░░░░░░░░░░│ 0.1234
#   轮2 (  6,400,000条) │██████████████░░░░░░░░░░░░░░░│ 0.1356
#   轮3 (  9,600,000条) │████████████████████░░░░░░░░░│ 0.1421
#   📈 模型从更多数据中显著受益
```

---

## 🧠 Algorithm

### Why Offline RL?

推荐本质是**序列决策** — 推荐什么会影响用户后续行为。传统的 (user, movie) → rating 监督学习只拟合历史，不学策略。

Online RL 需要实时交互环境，MovieLens 是静态快照 → 走 **Offline RL**：从历史轨迹中学策略。

### Why BCQ?

BCQ (Batch-Constrained Q-Learning, **Fujimoto et al., ICML 2019**) 解决 offline RL 的核心问题 — **distribution shift**。

标准 DQN 在 offline 数据上直接训练会崩溃：Q 网络对 "没出现过的 action" 给出荒谬高估值 → 策略选错。

BCQ 用 VAE 约束策略：

```
π(s) = argmax_{a ~ G(s)} Q(s, a + ξ(s, a))
                   ↑ VAE sampling   ↑ perturbation (|ξ| ≤ Φ)
```

只在 "历史行为覆盖的区域" 内选动作，从根本上避免外推错误。

### Components

| Component | Role | Interview Hook |
|-----------|------|----------------|
| **VAE Generator** | Learn behavior policy distribution, constrain candidates | "Distribution Shift 的数学解法" |
| **Perturbation ξ** | Small improvement within data support (clipped to Φ) | "有限探索 (Limited Exploration)" |
| **Q-Network (×2)** | Value estimation (Double Q prevents overestimation) | "Bellman Equation, TD Error" |
| **State Encoder (GRU)** | Encode rating history → state vector | "序列建模 vs 静态特征" |

---

## 📊 Interview Cheat Sheet

### Spark 八股 → Code Mapping

| Question | Where |
|----------|-------|
| Lazy Evaluation | `preprocess.py` — transformations build DAG, actions trigger compute |
| Wide vs Narrow Dependency | `preprocess.py` — `groupBy` → Shuffle (wide); `select`/`filter` → none (narrow) |
| Window Functions | `trajectories.py` — `PARTITION BY userId ORDER BY timestamp ROWS BETWEEN K PRECEDING` |
| Broadcast Join | `preprocess.py` — `F.broadcast()` small table to all executors |
| Data Skew | `trajectories.py` — analyze_user_activity() heavy-user warning |
| AQE | `config.py` — `spark.sql.adaptive.enabled = true` |
| Parquet | `preprocess.py` — columnar, predicate pushdown, schema self-contained |

### RL 八股 → Concept Mapping

| Question | Concept |
|----------|---------|
| MDP Modeling | State / Action / Reward / Transition definition |
| Bellman Equation | TD target: `r + γ · max Q'(s', a')` |
| Off-Policy vs Offline | Off-policy: learn from others' data; Offline: learn from fixed dataset |
| Distribution Shift | Why BCQ exists — Q extrapolates on unseen actions |
| Double DQN | Two Q-nets → min(Q1, Q2) prevents overestimation |
| Target Network | Polyak averaging: `θ' ← τ·θ + (1-τ)·θ'` |
| OPE | IPS (unbiased, high variance) / DM (biased, low variance) / DR (doubly robust) |

---

## 🔬 Ablation Study Design

```
BCQ Full:    VAE + Perturbation + Double Q
- VAE:       VAE → uniform sampling (measure constraint value)
- Perturb:   Φ = 0 (measure perturbation value)
- Double Q:  Single Q (measure overestimation)

Expected: BCQ > No-Perturb > No-VAE > Random
           No-VAE shows most severe Q-value inflation (distribution shift proof)
```

---

## 📦 References

- **BCQ**: Fujimoto et al., *"Off-Policy Deep Reinforcement Learning without Exploration"*, ICML 2019
- **MovieLens 25M**: GroupLens Research, https://grouplens.org/datasets/movielens/25m/
- **OPE**: Dudík et al., *"Doubly Robust Policy Evaluation and Learning"*, ICML 2011

---

## 👤 Author

**南志锦 (Nan Zhijin)** — 2026

> *"别人用协同过滤做推荐，我用强化学习。别人单机 Pandas，我 Spark 分布式。这个项目出来，面试官会记住我。"*

⭐ If this project helps your interview prep, give it a star!
