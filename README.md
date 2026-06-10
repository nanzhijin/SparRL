<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PySpark-4.1.2-orange?logo=apache-spark" alt="PySpark">
  <img src="https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/data-MovieLens%2025M-green" alt="MovieLens">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
  <img src="https://img.shields.io/badge/status-learning-blueviolet" alt="Learning">
</p>

<h1 align="center">SparRL</h1>
<h3 align="center">Spark 分布式推荐系统 + 强化学习技术探索</h3>

<p align="center">
  <b>工程主线：用 Spark 窗口函数处理 2500 万条评分数据，构建推荐模型训练管线。</b><br>
  <b>技术探索：跟着王喆《深度学习推荐系统 2.0》尝试用离线强化学习建模推荐决策。</b><br>
  <i>A Spark-based recommendation pipeline with an RL exploration side-quest.</i>
</p>

---

## 🎯 这个项目做了什么

**背景**：我在准备推荐算法岗位的面试。作为非科班转行的本科生，我需要一个能展示工程能力的项目——而不是一个"看起来很厉害但我自己都讲不清楚"的项目。

**工程主线（我真正掌握的部分）**：

| 做了什么 | 怎么做的 | 为什么值得讲 |
|----------|---------|------------|
| 数据管线 | PySpark 处理 25M 条评分 → parquet | 工业级数据量，不是 pandas 能扛的 |
| 特征工程 | genre multi-hot / year 分桶 / 统计特征 | 内容特征 + 协同特征的混合设计 |
| 轨迹构建 | 窗口函数 `PARTITION BY userId ORDER BY timestamp ROWS BETWEEN` | 把静态评分表变成可训练的序列样本 |
| 模型训练 | PyTorch DataLoader + GRU 序列编码 + 电影嵌入 | 端到端的训练管线 |
| 评估框架 | 时间窗口在线模拟 → NDCG 学习曲线 | 比单次 train/test split 多看一个维度 |
| Baseline | ItemCF（纯 pandas 手写）+ Popularity + Random | 没有对比就没有结论 |

**技术探索（我正在学习中的部分）**：

跟着**王喆老师《深度学习推荐系统 2.0》**的思路，我尝试把推荐建模成序列决策问题（MDP），用离线强化学习（BCQ）替代监督学习。目前的理解深度：知道为什么 RL 比监督学习更适合序列推荐场景，能讲清楚 Distribution Shift 的问题直觉，但 VAE 的数学推导和 OPE 的统计性质还在学习。

**我自己做的对比实验**：在同一个数据上跑了 BCQ 和 DQN，在线模拟中 BCQ 在数据稀疏的早期轮次表现更好——这验证了书里讲的"离线场景下无约束 Q-Learning 会外推错误"的理论。这个实验结果让我对 RL 在推荐领域的价值有了直观感受。

> ⚠️ **诚实声明**：这个项目的代码在 **Claude Code** 的辅助下完成。每个技术决策我过了一遍理解，但不是每个数学细节我都能手推。项目状态是"学习中"，不是"学完了"。

---

## 🏗 项目结构 & 文件详解

### 总览树

```
SparkPractice/                        # 总 ~4,700 行 Python
├── config.py                         # 全局配置中心 (149 行)
├── run_pipeline.py                   # 一键管线入口 (171 行)
├── requirements.txt                  # 依赖 (pyspark/torch/pandas/pyarrow)
├── itemcf_pandas.py                  # ItemCF 纯 pandas 手写 (368 行)
│
├── spark/                            # ━━━ Spark 数据层 ━━━
│   ├── __init__.py                   # 包标记 (空文件)
│   ├── preprocess.py                 # 特征工程 + 时间窗口切割 (737 行) ★
│   ├── trajectories.py               # 窗口函数轨迹构建 (468 行) ★★
│   └── ope.py                        # 离线策略评估 (337 行)
│
├── model/                            # ━━━ PyTorch 模型层 ━━━
│   ├── __init__.py                   # 包标记 (空文件)
│   ├── embeddings.py                 # 电影嵌入 + State Encoder (252 行)
│   ├── bcq.py                        # BCQ 完整模型 (455 行) ★★
│   ├── dqn.py                        # DQN 对比模型 (275 行)
│   └── train.py                      # 训练循环 + 在线模拟 (855 行) ★
│
├── eval/                             # ━━━ 评估层 ━━━
│   ├── __init__.py                   # 包标记 (空文件)
│   └── evaluate.py                   # 指标 + baseline + 学习曲线 (642 行)
│
├── data/ml-25m/ml-25m/               # MovieLens 25M 数据集 (需单独下载, ~250MB)
│   ├── ratings.csv                   #   25,000,095 条评分 (userId/movieId/rating/timestamp)
│   ├── movies.csv                    #   62,423 部电影 (movieId/title/genres)
│   ├── genome-scores.csv             #   1,558 个标签 × 电影 的相关性分数
│   ├── genome-tags.csv               #   1,128 个标签描述
│   ├── tags.csv                      #   1,093,360 条用户标签
│   └── links.csv                     #   MovieLens ID ↔ IMDb/TMDb ID 映射
│
└── output/                           # 运行时自动生成 (gitignore 排除)
    ├── movie_features.parquet        #   电影特征 (genre + year + stats)
    ├── user_features.parquet         #   用户特征 (偏好向量 + 行为统计)
    ├── ratings_train.parquet         #   训练集评分
    ├── ratings_test.parquet          #   测试集评分
    ├── trajectories.parquet          #   RL 训练轨迹 ★
    ├── bcq_model.pt                  #   BCQ 模型权重
    ├── time_windows_meta.json        #   时间窗口元数据
    └── online_windows/               #   在线模拟数据
        ├── round1_train.parquet      #     轮1训练 (W0)
        ├── round1_test.parquet       #     轮1测试 (W1)
        ├── round1_bcq_model.pt       #     轮1 BCQ 权重
        ├── round1_dqn_model.pt       #     轮1 DQN 权重
        ├── round2_*                  #     轮2 (W0+W1 训练 → W2 测试)
        └── round3_*                  #     轮3 (W0+W1+W2 训练 → W3 测试)
```

### 逐文件详解

---

#### ⚙️ `config.py` — 全局配置中心 (149 行)

整个项目的唯一配置入口。修改超参数、路径、开关都在这里。

| 配置块 | 关键变量 | 说明 |
|--------|---------|------|
| **路径** | `DATA_DIR`, `OUTPUT_DIR`, `TRAJECTORIES_PATH` | 所有文件路径集中管理 |
| **Spark** | `SPARK_CONFIG` dict | master/local[*]、内存 8g、AQE 自适应优化 |
| **数据过滤** | `MIN_USER_RATINGS=10`, `MIN_MOVIE_RATINGS=5` | 冷启动过滤阈值 |
| **轨迹** | `STATE_SIZE=10`, `ACTION_DIM=64` | state=最近10部电影, action=64维嵌入 |
| **特征工程** | `MOVIELENS_GENRES` (20个), `N_GENOME_TAGS=50` | genre multi-hot + genome tag 维度 |
| **BCQ 超参** | `HIDDEN_DIM=256`, `LATENT_DIM=32`, `PERTURBATION_PHI=0.05` | 模型结构 + VAE 隐空间 + 扰动幅度 |
| **DQN 超参** | `GAMMA=0.99`, `TAU=0.005`, `EPSILON_DECAY=0.995` | 折扣因子 + 软更新 + 探索衰减 |
| **训练** | `BATCH_SIZE=256`, `NUM_EPOCHS=50`, `VAE_LOSS_WEIGHT=0.5` | 训练控制 |
| **评估** | `EVAL_K_VALUES=[5,10,20]`, `RANDOM_SEED=42` | Top-K 评估配置 |
| **在线模拟** | `ONLINE_SIMULATION=True`, `N_TIME_WINDOWS=4` | 时间窗口开关 + 窗口数 |
| **AutoDL** | `detect_autodl()` 函数 | 自动检测 GPU + CUDA 环境 |

> 💡 **设计原则**：超参不散落在各文件中，改一个值全局生效。这是跟 Spark 的 `SparkConf` 学的集中管理模式。

---

#### 🚀 `run_pipeline.py` — 一键管线入口 (171 行)

给不想逐个跑脚本的人用的。四模式切换：

| 模式 | 命令 | 做了什么 |
|------|------|---------|
| `local` | `--mode local --sample 5000` | 采样 5000 用户 → Spark 管线 → 训练 → 评估，**10 分钟验证全流程** |
| `full` | `--mode full` | 全量 16 万用户 → 完整管线 → OPE |
| `spark-only` | `--mode spark-only` | 只跑 Spark 数据层（产出轨迹 parquet） |
| `train-only` | `--mode train-only` | 已有轨迹数据，只跑训练 |
| `eval-only` | `--mode eval-only` | 已有模型，只跑评估 |

每一步之间会自动检查上游文件是否存在，不存在会提示先跑哪一步。

> 💡 **设计原则**：把 end-to-end 的编排逻辑从具体脚本中抽出来。每个子脚本（preprocess/train/evaluate）也可以独立运行和调试。

---

#### 📊 `itemcf_pandas.py` — ItemCF 纯 pandas 手写 (368 行)

**6 个模块，从加载到解释，不用任何 ML 库**：

| 模块 | 做了什么 | 关键技术点 |
|------|---------|-----------|
| **模块0** | 加载 MovieLens 25M，采样 6000 用户 | 冷门电影过滤 (≥5 条评分) |
| **模块1** | `pivot_table` 构建 User-Item 矩阵 | 稀疏矩阵，NaN = 没评过 |
| **模块2** | **均值中心化** | `ui_matrix.subtract(user_means, axis=0)` — 消除用户评分偏置 |
| **模块3** | **余弦相似度** | `normalized @ normalized.T` — NumPy 向量化，O(n²) 但秒出 |
| **模块4** | **共同评分用户数过滤** | `binary @ binary.T` — 点积算共现次数，<3 的过滤 |
| **模块5** | **为具体用户生成推荐** | 加权评分 = `Σ(sim × rating) / Σ(sim)`，排除已看电影 |
| **模块6** | **可解释性追溯** | 每部推荐电影 → 追溯"因为用户喜欢 X，而 X 和 Y 相似" |

整个文件是推荐系统基础概念的动手实现——协同过滤、相似度、去偏置、可解释性，都在纯 NumPy/Pandas 里展示。

> 🎯 **面试价值**：这是"我知道推荐系统底层在算什么"的证明。不是调 `surprise` 库，是手写矩阵乘法。

---

#### 🔧 `spark/preprocess.py` — Spark 特征工程 (737 行)

**6 个模块，产出 4 个 parquet 文件**：

| 模块 | 做了什么 | Spark 考点 |
|------|---------|-----------|
| **模块0** | `SparkSession.builder` 初始化 | local[*]、AQE 自适应优化、内存配置 |
| **模块1** | 加载 `ratings.csv` + `movies.csv` | 显式 schema（生产最佳实践）、`cache()` 物化 |
| **模块2** | 过滤冷用户 (<10 评分) + 冷电影 (<5 评分) | `groupBy` → `filter` → `broadcast join` |
| **模块3** | **电影特征工程** | `split`/`regexp_extract`（内置函数 > UDF）、`percent_rank()` 窗口函数 |
| **模块4** | **用户特征工程** | `groupBy` 聚合 + genre 偏好向量 (20 维) |
| **模块5a** | 训练/测试按时序划分 (前 80% vs 后 20%) | `ROW_NUMBER() OVER (PARTITION BY userId ORDER BY timestamp)` |
| **模块5b** | **全局时间窗口切割** ★ | `cache()` 复用 + 逐窗口 `filter` + 元数据 JSON |
| **模块6** | 保存 parquet + 打印执行计划 | `explain()` 查看 DAG |

```
模块3 电影特征产出:
  genre_0..19:    multi-hot (20 维)
  release_year:   分桶嵌入 (1900-2030, 每10年一桶)
  avg_rating:     全局平均评分
  rating_std:     评分标准差 (争议度)
  rating_count:   评分次数
  popularity_percentile: 热度分位数 (0~1)

模块4 用户特征产出:
  avg_rating:     该用户平均评分 (手松/手紧)
  rating_std:     评分区分度
  rating_count:   评分总数
  active_days:    时间跨度
  pref_0..19:     各 genre 偏好 (genre 电影的评分均值)
```

> 🎯 **面试价值**：Spark 八股全在这一个文件里——Lazy Evaluation、宽窄依赖、Broadcast Join、AQE、Parquet 列存。

---

#### ⭐ `spark/trajectories.py` — Spark 窗口函数轨迹构建 (468 行)

**整个项目的 Spark 技术制高点**。把 2500 万条评分变成 500 万条 RL 训练轨迹。

```
核心 SQL 等价逻辑:

WITH ordered AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY userId ORDER BY timestamp
  ) AS rn
  FROM ratings_train
),
with_state AS (
  SELECT *,
    COLLECT_LIST(STRUCT(movieId, rating)) OVER (
      PARTITION BY userId ORDER BY rn
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING    ← state: 前10部
    ) AS state_structs,
    COLLECT_LIST(STRUCT(movieId, rating)) OVER (
      PARTITION BY userId ORDER BY rn
      ROWS BETWEEN 9 PRECEDING AND CURRENT ROW      ← next_state: 包含当前
    ) AS next_state_structs
  FROM ordered
)
SELECT * FROM with_state WHERE rn > 10  ← 过滤不完整state
```

| 模块 | 做了什么 | 关键技术点 |
|------|---------|-----------|
| **模块0** | 初始化 SparkSession | 同 preprocess |
| **模块1** | 加载训练集评分 (parquet) | 谓词下推 |
| **模块2** | **窗口函数构建轨迹** ★ | `PARTITION BY userId ORDER BY timestamp`, `ROWS BETWEEN K PRECEDING` |
| **模块3** | 轨迹质量分析 | Reward 分布、state 长度分布、数据倾斜检测 (p99 vs max) |
| **模块4** | PyTorch 数据准备 | Reward 归一化 [0.5,5.0]→[0,1]、`coalesce` 控制分区 |
| **模块5** | 保存 parquet + 展示执行计划 | `explain()` 查看 Window→Project→Filter DAG |
| **模块6** (Online) | 逐轮轨迹构建 | `--online` 模式：读取窗口数据，逐轮构建 + 保存元信息 |

> 🎯 **面试价值**：PARTITION BY 不触发 Shuffle（数据已按 userId 分好区）、ROWS 模式内存 O(K)、collect_list + struct 保持关联。

---

#### 📐 `spark/ope.py` — 离线策略评估 (337 行)

不部署上线也能估计策略效果——面试差异化武器。在 Spark 上分布式计算三种估计器：

| 估计器 | 思路 | 公式 | 优点 | 缺点 |
|--------|------|------|------|------|
| **IPS** | 重要性采样重加权 | `V = avg(w_i · r_i)`, w = π_e/π_b | 无偏 | 高方差 |
| **DM** | Reward model 直接估 | `V = avg(r̂(s_i, a_i))` | 低方差 | 有偏 |
| **DR** | IPS + DM 双保险 | `V = V_DM + avg(w_i · (r_i - r̂_i))` | 任一 model 对就无偏 | 需要两个 model |

简化实现：用评分频率作为 propensity、用全局平均分作为 reward model。实际替换为 BCQ 的 Q 网络即可做完整 OPE。

> 🎯 **面试价值**："只要 reward model 或 propensity model 有一个对，DR 就是无偏的"——这体现了对评估方法论的统计学理解。

---

#### 🧩 `model/embeddings.py` — 电影嵌入 + State Encoder (252 行)

三个类，一条嵌入管线：

| 类 | 输入 | 输出 | 关键设计 |
|----|------|------|---------|
| **MovieEmbedding** | genre(20) + year(1) + stats(3) | 64-dim 稠密向量 | content+collaborative 混合，冷门电影也有合理嵌入 |
| **MovieEmbeddingLookup** | movieId | 64-dim 嵌入 | 预计算全部电影嵌入 → `register_buffer` 不参与梯度 → O(1) 索引 |
| **StateEncoder** | 最近K部电影嵌入 + 对应评分 | 128-dim state 向量 | 双向 GRU，最后拼接 forward+backward hidden |

```
电影嵌入的三个通道:
  genre (20-dim multi-hot) → Linear(20→32) → ReLU → Linear(32→32)
  year  (scalar)           → Embedding(13→8) 分桶查表
  stats (avg/std/pop)      → Linear(3→16) → ReLU
                              ↓
                          concat [32+8+16=56] → Linear(56→64) → LayerNorm → 64-dim 输出
```

> 🎯 **面试价值**：Content+Collaborative 混合嵌入 → 冷启动友好；GRU 捕获时序依赖（"刚看完恐怖片 vs 刚看完喜剧"影响后续推荐）。

---

#### 🧠 `model/bcq.py` — BCQ 完整模型 (455 行)

**SparRL 的 RL 算法核心**，四个类组合：

| 类 | 角色 | 结构 | 面试考点 |
|----|------|------|---------|
| **VAEGenerator** | 行为约束 | Encoder(s,a)→(μ,σ²) → z~N(μ,σ) → Decoder(s,z)→â | 条件 VAE p(a\|s)，不是在生成，是在约束 |
| **Perturbation** | 有限改进 | MLP(s,a)→ξ → a+Φ·tanh(ξ)，\|ξ\|≤Φ | Φ 越小越保守，Φ=0 退化为纯 VAE |
| **QNetwork** | 价值评估 | MLP(s,a)→scalar Q 值 | 评估"在状态 s 推荐电影 a 的长期收益" |
| **BCQ** | 完整策略 | VAE + Perturb + Double Q | π(s)=argmax Q(s, a'+ξ(s,a')) |

```
BCQ 推理流程 (select_action):

  State s ──→ VAE.sample(s, n=100) ──→ {a₁, a₂, ..., a₁₀₀} 候选动作
                  │
                  ├──→ Perturbation(s, aᵢ) ──→ {a₁', a₂', ..., a₁₀₀'} 微调后
                  │
                  └──→ Q1(s, aᵢ') ──→ {q₁, q₂, ..., q₁₀₀} Q 值打分
                                          │
                                     argmax → 最佳动作嵌入
                                          │
                              最近邻搜索 → 具体电影 ID
```

BCQ 训练三路损失：

| 损失 | 公式 | 作用 |
|------|------|------|
| **Q Loss** | `MSE(Q(s,a), r + γ·Q_target(s',a'))` | 学 Bellman 最优方程 |
| **VAE Loss** | `MSE(a, â) + β·KL(N(μ,σ²)‖N(0,1))` | 学行为策略分布 |
| **Perturb Loss** | `-Q(s, a+ξ(s,a))` | 在约束内最大化 Q 值 |

> 🎯 **面试价值**：Distribution Shift 的数学解法——VAE 只让策略在"历史上出现过"的区域选动作。

---

#### 🎮 `model/dqn.py` — DQN 对比模型 (275 行)

跟 BCQ 用相同的 Q 网络架构，但**去掉 VAE 约束**——唯一的变量就是"有没有行为约束"。

| 组件 | 实现 | 作用 |
|------|------|------|
| **QNetwork** | 复用 `bcq.py` 的 QNetwork（双 Q+目标网络） | 控制变量，保证架构一致 |
| **Replay Buffer** | `deque(maxlen=100000)` | 打破序列相关性 |
| **ε-greedy** | `ε=0.5→0.01, decay=0.995` | 训练时探索，评估时纯贪婪 |
| **Candidate Actions** | top-2000 热门电影嵌入 | 离散候选集做 Q 值打分 |
| **DQNTrainer** | `fill_replay_buffer → train_epoch(×N) → soft_update` | 标准 DQN 训练循环 |

```
BCQ vs DQN 唯一的区别:
  BCQ:  VAE 生成候选 → Perturb 微调 → Q 选最佳   (constrained)
  DQN: ε-greedy 在所有候选里选                    (unconstrained)
  
  → 如果 BCQ > DQN，就是 VAE 约束的功劳
  → 这就是 controlled experiment
```

> 🎯 **面试价值**：控制变量归因——科学方法论。不是"BCQ 比 DQN 好"这种模糊结论，而是"VAE 约束具体贡献了多少"。

---

#### 🏋️ `model/train.py` — 训练循环 (855 行)

项目最长的文件，管理整个训练生命周期：

| 组件 | 职责 |
|------|------|
| **TrajectoryDataset** | 从 parquet 加载轨迹数据 → PyTorch Dataset |
| **build_movie_lookup** | 从电影特征 parquet → MovieEmbeddingLookup 全量嵌入表 |
| **collate_batch** | 自定义 batch 拼装：movieId→嵌入→GRU→state 向量 |
| **BCQTrainer** | VAE 预训练 Phase A + BCQ 联合训练 Phase B + checkpoint |
| **DQNTrainer** | Replay Buffer 填充 + ε-greedy 训练 + 候选动作管理 |

```
BCQ 训练两阶段:
  Phase A (VAE 预训练, 5-10 epochs):
    只训练 VAE + StateEncoder + MovieLookup
    目标: VAE 学会"给定 state，复现 action 的分布"
    原因: Q 网络在 VAE 收敛前参与训练会导致梯度不稳定

  Phase B (联合训练, 50 epochs):
    三路优化器交替更新:
      optimizer_q:       Q1 + Q2 → 最小化 TD 误差
      optimizer_vae:     VAE + StateEncoder → 保持 action 约束
      optimizer_perturb: Perturbation → 在约束内最大化 Q 值
    + CosineAnnealingLR 学习率调度
    + 混合精度 AMP (CUDA 时自动启用)
    + Polyak 软更新目标网络 (每 batch)
```

**在线模拟模式** (`--online --algo both`)：逐轮加载轨迹 → 训练 → 保存 checkpoint → 下一轮累积数据继续训练。

> 🎯 **面试价值**：AMP 混合精度、cosine annealing、分组优化器、Polyak 软更新——训练工程细节体现 PyTorch 熟练度。

---

#### 📈 `eval/evaluate.py` — 模型评估 (642 行)

**两套评估逻辑**：

**标准评估** — 在测试集上对比所有方法：

| 指标 | 公式 | 面试考点 |
|------|------|---------|
| **NDCG@K** | `DCG/IDCG`，DCG=Σ(2^rel-1)/log₂(pos+1) | 最全面——考虑排序+相关性 |
| **Recall@K** | Top-K 中命中相关/全部相关 | 最直观——但不对排序敏感 |
| **Precision@K** | Top-K 中相关占比 | 配合 Recall 看 Precision-Recall tradeoff |
| **MRR** | 第一个相关结果的倒数排名 | "第一个推荐最重要"的场景 |

Baseline 对比：BCQ vs DQN vs ItemCF vs Popularity vs Random。

**在线学习曲线** (`--online --algo both`)：

```
轮1: 训练=W0      测试=W1   (数据最少)
轮2: 训练=W0+W1   测试=W2   (数据增长)
轮3: 训练=W0+W1+W2 测试=W3  (逼近全量)

输出:
  ╔════════════════════════════════════════════════╗
  ║  BCQ vs DQN — Online Learning Curve (NDCG@10) ║
  ╚════════════════════════════════════════════════╝
  轮   BCQ        DQN        Δ(BCQ-DQN)    诊断
  1    0.1234     0.0891     +0.0343       📗 BCQ wins
  2    0.1356     0.1201     +0.0155       📗 BCQ wins
  3    0.1421     0.1389     +0.0032       📙 tie
```

还能画 ASCII 学习曲线柱状图，一眼看出趋势。

> 🎯 **面试价值**：NDCG 最全面（面试重点讲）、Recall 最直观、MRR 适合"首推最重要"场景。学习曲线 = 数据效率分析。

---

### 文件规模总览

```
spark/preprocess.py     ████████████████████ 737 行  (数据管线主引擎)
spark/trajectories.py   ██████████████ 468 行        (Spark 技术制高点)
spark/ope.py            █████████ 337 行             (OPE 三件套)
model/embeddings.py     ███████ 252 行               (嵌入 + 状态编码)
model/bcq.py            ████████████ 455 行           (RL 算法核心)
model/dqn.py            ████████ 275 行               (DQN 对比线)
model/train.py          ██████████████████████ 855 行 (训练循环最复杂)
eval/evaluate.py        █████████████████ 642 行      (评估 + 学习曲线)
config.py               ████ 149 行                   (配置中心)
run_pipeline.py         ████ 171 行                   (一键入口)
itemcf_pandas.py        ██████████ 368 行             (手写 ItemCF)
requirements.txt        █ 10 行                       (依赖)
─────────────────────────────────────────────
总计                    4,719 行 Python
```

---

## 🚀 运行

```bash
pip install -r requirements.txt

# 本地验证 (5K 用户, ~10 分钟)
python run_pipeline.py --mode local --sample 5000

# 在线模拟 — 时间窗口增量学习
python run_pipeline.py --mode spark-only
python spark/trajectories.py --online
python model/train.py --online --algo both    # BCQ + DQN 对比
python eval/evaluate.py --online --algo both  # 学习曲线
```

---

## 📖 学习旅程（真实记录）

### 起点：推荐不就是协同过滤吗？

项目一开始我用纯 pandas 写了个 ItemCF（`itemcf_pandas.py`）。评分矩阵 → 均值中心化 → 余弦相似度 → 加权推荐。整个过程没有损失函数、没有梯度、没有任何 ML 库——只有线性代数和排序。

这让我对"推荐系统最基本的信号来自哪里"有了直观感受：**共现**。两部电影被同一群人喜欢，它们就有协同关系。

### 转折：看了王喆书，意识到推荐是序列决策

王喆老师在书里反复强调一个观点：**推荐不是"猜你喜欢什么"，而是"在这个时间点推什么能让你继续用下去"**。前者是监督学习（拟合历史），后者是序列决策（影响未来）。

这让我开始思考：如果推荐会影响用户的后续行为，那每个推荐决策就需要考虑长期收益。这正是强化学习的框架——state（用户当时的兴趣状态）、action（推荐什么）、reward（用户反馈）、next_state（看完后的新状态）。

### 尝试：用 Spark 窗口函数做范式转换

把 2500 万条静态评分变成 RL 训练轨迹，是工程上最考验人的一步。

```sql
-- 核心逻辑（Spark 窗口函数）
SELECT userId, timestamp,
  COLLECT_LIST(STRUCT(movieId, rating)) OVER (
    PARTITION BY userId ORDER BY timestamp
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
  ) AS state,                                          -- 最近10部 = 当前兴趣状态
  movieId AS action,                                   -- 当前推荐 = 决策
  rating AS reward,                                    -- 用户评分 = 反馈
  COLLECT_LIST(STRUCT(movieId, rating)) OVER (
    PARTITION BY userId ORDER BY timestamp
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS next_state                                      -- 看完后 = 新状态
FROM ratings_train
```

这一步体现了 Spark 窗口函数在推荐场景的真正价值：把一个"监督学习的数据集"转换成"序列决策的数据集"，全程分布式，2500 万条数据分钟级出结果。

### 发现：BCQ 和 DQN 在在线模拟里的差异

我做了 BCQ vs DQN 的在线模拟对比——在 Claude Code 的帮助下实现了两个模型，然后用完全相同的数据和评估流程跑对比。结果：

```
轮1 (25%数据):  BCQ > DQN  差距明显  ← VAE约束在稀疏数据时最有效
轮2 (50%数据):  BCQ > DQN  差距缩小
轮3 (75%数据):  BCQ ≈ DQN  逐渐拉平  ← 数据够了，无约束DQN也能学好
```

这个实验让我直观地感受到了书里说的 **Distribution Shift** 问题：当数据不足以覆盖动作空间时，没约束的 Q-Learning 会对"没见过的电影"给出虚高估值，导致推荐质量下降。BCQ 的 VAE 约束就是为这个问题设计的。

### 当前状态：知道方向，还在填细节

我现在能讲清楚的部分：
- ✅ 为什么推荐适合建模成序列决策
- ✅ Distribution Shift 的直觉和为什么需要约束
- ✅ BCQ 和 DQN 在离线场景的对比意义
- ✅ Spark 窗口函数做轨迹构建的工程实现
- ✅ 在线模拟评估框架的设计思路

我还在学/需要补的部分：
- 📖 VAE 的 ELBO 推导和重参数化技巧的数学细节
- 📖 OPE 估计器的统计性质（IPS 的无偏性证明、DR 的双重鲁棒性理论）
- 📖 BCQ vs CQL vs BEAR 的算法选择依据——目前只知道 BCQ 的约束逻辑最直观
- 📖 HSTU（Meta GR）的无 Softmax 注意力机制——这是 Generative Recommender 的前沿方向

---

## 🔬 消融实验思路

我不是在调参优化最高指标，而是想验证"BCQ 的每个组件各自贡献了什么"：

```
BCQ Full:    VAE + Perturbation + Double Q
- VAE:       VAE → uniform sampling    (去掉行为约束)
- Perturb:   Φ = 0                     (去掉有限改进)
- Double Q:  Single Q                  (去掉防过估计)

预期: BCQ > No-Perturb > No-VAE > Random
      No-VAE 应该出现最严重的 Q 值虚高 → Distribution Shift 的实证
```

---

## 📊 面试叙事（诚实版）

当面试官问"这个项目你做了什么"，我会这样讲：

> "我做了两件事。第一是 Spark 工程——用窗口函数把 MovieLens 25M 数据构建成推荐模型的训练管线，包括特征工程、轨迹构建、在线模拟评估。这部分是我独立完成的。
>
> 第二是跟着王喆老师的书做了一个 RL 技术探索——把推荐建模成序列决策，在 Claude Code 的辅助下实现了 BCQ 和 DQN 的对比实验。我目前对 RL 的理解程度是知道'为什么需要约束'、'各组件起什么作用'，但数学细节还在补。这个部分对我更多是一个学习过程。"

如果面试官追问 BCQ 的技术细节：

| 问题 | 我的诚实回答方向 |
|------|----------------|
| "为什么用 BCQ？" | 王喆书里提到离线场景的核心问题是分布偏移，BCQ 的 VAE 约束是解决这个的经典方案。我对比了 BCQ 和 无约束 DQN，实验上也验证了约束的价值 |
| "VAE 的 ELBO 是什么？" | 这是我在补的数学基础部分——目前知道 VAE 通过编码-解码学习历史行为的分布，但 ELBO 的推导我能讲直觉还不能手推 |
| "为什么不用 CQL？" | 我知道 CQL 是另一个离线 RL 方案，但我还没读那篇论文，没法做技术对比 |
| "Spark 窗口函数怎么写的？" | 随时可以展开 PARTITION BY + ROWS BETWEEN 的完整逻辑 ← 这是我真的掌握的 |

> 核心原则：**工程部分——我做了，我能讲，你可以深挖。RL 部分——我在学，我试了，我有实验结论，但数学细节还在路上。不装。**

---

## 📦 参考 & 致谢

- **王喆**：《深度学习推荐系统 2.0》，电子工业出版社，2025 —— 这个项目 RL 部分的理论来源
- **BCQ**: Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration", ICML 2019
- **MovieLens 25M**: GroupLens Research
- **OPE**: Dudík et al., "Doubly Robust Policy Evaluation and Learning", ICML 2011
- **Claude Code**: Anthropic —— 代码生成、架构建议、技术文档辅助
- **ItemCF 参考**: 王喆《深度学习推荐系统》第2章

---

## 👤 作者

**南志锦 (Nan Zhijin)** — 2026

> *本科，转行算法。Spark + 推荐是我真的会的东西；RL 是我在学、在试、在思考的东西。这个 README 如实记录了一个学习过程，而不是一个虚假的"成品"。*
