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

## 🏗 项目结构

```
SparkPractice/
├── config.py                  # 全局配置
├── run_pipeline.py            # 端到端管线 (local/full/spark-only/train-only)
├── requirements.txt
├── itemcf_pandas.py           # ItemCF baseline (纯 pandas，0 ML 库)
│
├── spark/                     # Spark 数据管线 ★ 工程核心
│   ├── preprocess.py          # 特征工程 + 时间窗口切割 + 数据输出
│   ├── trajectories.py        # 窗口函数轨迹构建 ★ 核心 Spark 代码
│   └── ope.py                 # 离线策略评估 (IPS/DM/DR)
│
├── model/                     # PyTorch 模型
│   ├── embeddings.py          # 电影嵌入 + GRU State Encoder
│   ├── bcq.py                 # BCQ 模型 (VAE + Perturbation + Double Q)
│   ├── dqn.py                 # DQN 对比模型 (ε-greedy + Replay Buffer)
│   └── train.py               # 训练循环 + 在线模拟模式
│
├── eval/
│   └── evaluate.py            # NDCG/Recall/MRR + 在线学习曲线
│
├── data/ml-25m/ml-25m/        # MovieLens 25M (需下载)
└── output/                    # 中间数据 & 模型
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
