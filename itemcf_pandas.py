"""
ItemCF 协同过滤 — 纯 pandas 手写，从零拆解推荐系统地基
================================================================
南志锦 · 2026-06-08

不依赖任何 ML 库，每一步中间结果都可以 print 出来看。
数据集: MovieLens 25M, 取子集(前 3000 用户)保证单机跑得快。

核心理念:
  推荐 = 数数 + 加权
  "喜欢A的人也喜欢B" -> 统计共现 -> 相似度 -> 加权推荐

阅读顺序: 按模块编号 0->1->2->3->4->5->6 逐步执行
"""

import sys
import io
import pandas as pd
import numpy as np
from collections import defaultdict
import time

# 强制 stdout 用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# 模块0 — 数据加载 & 采样
# ============================================================
print("\n" + "=" * 60)
print("模块0 — 数据加载 & 采样")
print("=" * 60)

DATA_DIR = "data/ml-25m/ml-25m"

ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")

print(f"全量评分: {len(ratings):,} 行")
print(f"用户数:   {ratings['userId'].nunique():,}")
print(f"电影数:   {ratings['movieId'].nunique():,}")

# 取前 3000 个用户, 保证单机流畅跑
# 同时只保留有足够评分的电影 (>= 5 条), 减少噪声
sample_users = ratings['userId'].unique()[:6000]
df = ratings[ratings['userId'].isin(sample_users)].copy()

# 过滤冷门电影: 评分数 < 5 的去掉
movie_counts = df.groupby('movieId').size()
popular_movies = movie_counts[movie_counts >= 5].index
df = df[df['movieId'].isin(popular_movies)]

print(f"采样用户:   {len(sample_users):,}")
print(f"采样评分:   {len(df):,} 行")
print(f"采样电影:   {df['movieId'].nunique():,}")
sparsity = len(df) / (len(sample_users) * df['movieId'].nunique())
print(f"稀疏度:     {sparsity:.4f} (即矩阵里只有 {sparsity * 100:.2f}% 有值)")

# ============================================================
# 模块1 — 构建用户-电影评分矩阵
# ============================================================
print("\n" + "=" * 60)
print("模块1 — 用户-电影评分矩阵 (User-Item Matrix)")
print("=" * 60)
print("""
这是协同过滤的核心数据结构:
  - 行 = 用户 (userId)
  - 列 = 电影 (movieId)
  - 值 = 评分 (rating)
  - 空白 = 没评过 (NaN)

大部分格子是空的 — 这就是"稀疏矩阵"。
协同过滤的魔力: 用已有的值推断空白该填多少。
""")

# pivot 构建矩阵
t0 = time.time()
ui_matrix = df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    aggfunc='mean'  # 万一有重复取均值
)
elapsed = time.time() - t0

print(f"矩阵形状: {ui_matrix.shape[0]:,} 用户 x {ui_matrix.shape[1]:,} 电影")
print(f"构建耗时: {elapsed:.2f}s")

# 看一眼矩阵
print("\n矩阵前 5 行 x 前 8 列 (NaN = 没评分):")
print(ui_matrix.iloc[:5, :8].to_string())
print(f"\n用户 1 评过的电影数: {ui_matrix.loc[1].notna().sum()}")
print(f"用户 1 平均评分:        {ui_matrix.loc[1].mean(skipna=True):.2f}")
print(f"用户 1 评分值分布:")
print(ui_matrix.loc[1].dropna().value_counts().sort_index())

# ============================================================
# 模块2 — 均值中心化（去偏差）
# ============================================================
print("\n" + "=" * 60)
print("模块2 — 均值中心化 (Mean Centering)")
print("=" * 60)
print("""
为什么需要这一步?
  - 有人打分手松(平均 4.5), 有人手紧(平均 2.5)
  - 不处理的话, "手松用户的 3 分" 和 "手紧用户的 3 分" 不是一回事
  - 减去每个用户的均值后: 正数 = "比他自己平均高", 负数 = "比他自己平均低"
""")

# 每行(每个用户)减去自己的均值
user_means = ui_matrix.mean(axis=1, skipna=True)
print(f"用户均值范围: {user_means.min():.2f} ~ {user_means.max():.2f}")

ui_centered = ui_matrix.subtract(user_means, axis=0)

print("\n中心化后矩阵前 5 行 x 前 5 列:")
print(ui_centered.iloc[:5, :5].to_string())
print("""
解读: 正数 = 比该用户平均分高(真喜欢)
      负数 = 比该用户平均分低(不太喜欢)
      NaN  = 没评过
""")

# ============================================================
# 模块3 — 计算电影-电影相似度 (Item-Item Similarity)
# ============================================================
print("\n" + "=" * 60)
print("模块3 — 电影-电影相似度矩阵")
print("=" * 60)
print("""
这里用 余弦相似度 (Cosine Similarity), 公式:

  sim(A,B) = sum_u[(r_ua - mu_u) * (r_ub - mu_u)] / (|r_a| * |r_b|)

  分子: 两部电影评分偏差的协方差("一起高一起低"的程度)
  分母: 各自偏差幅度的乘积(归一化, 防大数吃小数)

用人话: 两部电影被同一群用户评分的模式有多像。
  sim = 1.0  -> 完全一致(喜欢A也喜欢B, 讨厌A也讨厌B)
  sim = 0.0  -> 毫无关系
  sim = -1.0 -> 完全相反(喜欢A的都讨厌B)

只计算"至少被 3 个共同用户评过"的电影对 -> 减少偶然性
""")

# 用 numpy 计算, 比 pandas 循环快得多
# center 后的矩阵转 numpy, NaN 填 0
# 转置: 每行 = 一部电影的评分向量 (n_movies, n_users)
matrix = ui_centered.fillna(0).values.T

norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # 每部电影的 L2 范数
norms[norms == 0] = 1e-10  # 避免除 0

# 归一化: 每部电影变成单位向量
normalized = matrix / norms

# 余弦相似度 = 归一化向量的点积
# (n_movies, n_movies)
print(f"计算 {matrix.shape[0]:,} x {matrix.shape[0]:,} 相似度矩阵...")
t0 = time.time()

similarity = normalized @ normalized.T  # 余弦相似度矩阵
elapsed = time.time() - t0
print(f"相似度矩阵计算耗时: {elapsed:.2f}s")
print(f"矩阵形状: {similarity.shape}")

# 转为 pandas, 带电影 ID
movie_ids = ui_matrix.columns.tolist()
sim_df = pd.DataFrame(similarity, index=movie_ids, columns=movie_ids)

# 对角线是自己跟自己比, 不推荐自己
np.fill_diagonal(similarity, np.nan)

print(f"\n相似度矩阵片段 (movieId 1 ~ 8):")
print(sim_df.iloc[:8, :8].round(3).to_string())

# 找出最相似的电影对
print("\n--- 最相似的电影对 ---")
# stack -> (movieA, movieB) 长表
sim_long = sim_df.stack().reset_index()
sim_long.columns = ['movieA', 'movieB', 'similarity']
# 只取上三角避免重复 (A,B) 和 (B,A) 各算一次
sim_long = sim_long[sim_long['movieA'] < sim_long['movieB']]
sim_long = sim_long.sort_values('similarity', ascending=False)

# 关联电影名
movie_names = movies.set_index('movieId')['title']
sim_long['titleA'] = sim_long['movieA'].map(movie_names)
sim_long['titleB'] = sim_long['movieB'].map(movie_names)

print("\nTop-15 最相似电影对:")
for i, (_, row) in enumerate(sim_long.head(15).iterrows(), 1):
    print(f"  {i:>2}. {row['titleA']:<45} <<-->> {row['titleB']:<45}  sim={row['similarity']:.4f}")

# ============================================================
# 模块4 — 共同评分用户数检查
# ============================================================
print("\n" + "=" * 60)
print("模块4 — 共同评分用户数 (Co-rating Count)")
print("=" * 60)
print("""
相似度高不一定可靠 — 如果只有 2 个用户同时评过两部电影,
相似度即使为 1.0 也可能是偶然。

所以我们要检查"共同评分用户数"作为置信度指标。
""")

# 计算共现矩阵: 两部电影有多少共同评分用户
# 巧妙做法: 用 0/1 矩阵(评过=1, 没评过=0)的点积算共现次数
binary = ui_centered.notna().astype(int).values.T  # 评过 = 1, 没评过 = 0
co_rating = binary @ binary.T  # 点积 = 共同评过的用户数
np.fill_diagonal(co_rating, 0)

co_rating_df = pd.DataFrame(co_rating, index=movie_ids, columns=movie_ids)

print(f"\n共现矩阵片段 (movieId 1 ~ 8):")
print(co_rating_df.iloc[:8, :8].to_string())

# 过滤: 只保留共同评分 >= 3 的相似度
valid_mask = co_rating >= 3
sim_filtered = sim_df.where(valid_mask)

print(f"\n过滤前有效相似对: {sim_df.notna().sum().sum():,}")
print(f"过滤后有效相似对 (共同用户>=3): {sim_filtered.notna().sum().sum():,}")

# ============================================================
# 模块5 — 为具体用户生成推荐
# ============================================================
print("\n" + "=" * 60)
print("模块5 — 为具体用户生成推荐 (Generate Recommendations)")
print("=" * 60)
print("""
推荐逻辑(用人话):
  1. 找出用户 A 评分高的电影 (>= 他自己平均分的电影)
  2. 对每一部他喜欢的电影, 找出它的相似电影
  3. 对所有相似电影, 按相似度加权打分:
       预测分 = sum(相似度 x 用户对已看电影的评分) / sum(相似度)
  4. 排除他已经看过的
  5. 从高到低排序 -> Top-K
""")

TARGET_USER = 1
print(f"\n目标用户: userId = {TARGET_USER}\n")

# 用户的原始评分
user_ratings = ui_matrix.loc[TARGET_USER].dropna()
user_mean = user_means.loc[TARGET_USER]
print(f"该用户评过 {len(user_ratings)} 部电影, 平均分 {user_mean:.2f}")
print("评分最高的 10 部:")
high_rated = user_ratings[user_ratings >= user_mean].sort_values(ascending=False)
for mid, r in high_rated.head(10).items():
    title = movie_names.get(mid, f"Unknown({mid})")
    print(f"  [{mid:>6}] {title:<50}  {r:.1f}")

# --- 核心推荐逻辑 ---
recommendations = defaultdict(lambda: {'weighted_sum': 0.0, 'sim_sum': 0.0})

# 只看用户喜欢的电影 (评分 >= 他自己的均值)
liked_movies = user_ratings[user_ratings >= user_mean]

for liked_movie, rating in liked_movies.items():
    # 找这部喜欢电影的相似电影
    if liked_movie not in sim_df.columns:
        continue
    similar_to_liked = sim_df[liked_movie].dropna()

    for similar_movie, sim_score in similar_to_liked.items():
        # 排除用户已经看过的
        if similar_movie in user_ratings.index:
            continue
        # 相似度加权累加
        recommendations[similar_movie]['weighted_sum'] += sim_score * rating
        recommendations[similar_movie]['sim_sum'] += sim_score

# 计算预测分 = 加权和 / 相似度和
rec_results = []
for movie_id, vals in recommendations.items():
    if vals['sim_sum'] > 0:
        predicted_rating = vals['weighted_sum'] / vals['sim_sum']
        rec_results.append({
            'movieId': movie_id,
            'predicted_rating': predicted_rating,
            'sim_sources': vals['sim_sum'],  # 有多少相似"证据"
        })

rec_df = pd.DataFrame(rec_results).sort_values('predicted_rating', ascending=False)
rec_df['title'] = rec_df['movieId'].map(movie_names)

print(f"\n为该用户生成了 {len(rec_df)} 个推荐候选")
print("\n" + "-" * 40)
print("Top-20 推荐结果:")
print("-" * 40)
for i, (_, row) in enumerate(rec_df.head(20).iterrows(), 1):
    print(f"  {i:>2}. {row['title']:<50}  "
          f"预测分={row['predicted_rating']:.3f}  "
          f"证据={row['sim_sources']:.1f}")

# ============================================================
# 模块6 — 结果洞察：为什么推荐这些？
# ============================================================
print("\n" + "=" * 60)
print("模块6 — 结果洞察 (Why These Movies?)")
print("=" * 60)
print("""
推荐系统的"可解释性"就藏在这里 —
每部推荐电影背后都有"证据链": 因为用户喜欢 X, 而 X 和 Y 相似。
""")

# 取推荐第一名的电影, 追溯它是怎么来的
top_rec = rec_df.iloc[0]
top_movie_id = top_rec['movieId']
top_title = top_rec['title']

print(f"\n追溯: 为什么推荐了 [{top_title}] ?\n")

# 找出是哪些"已喜欢电影"贡献了最多的相似度
contributors = []
for liked_movie in liked_movies.index:
    if liked_movie in sim_df.index and top_movie_id in sim_df.columns:
        sim = sim_df.loc[liked_movie, top_movie_id]
        if not pd.isna(sim) and sim > 0:
            contributors.append({
                'liked_movie': liked_movie,
                'liked_title': movie_names.get(liked_movie, f"ID:{liked_movie}"),
                'rating': user_ratings[liked_movie],
                'similarity': sim,
                'contribution': sim * user_ratings[liked_movie],
            })

contrib_df = pd.DataFrame(contributors).sort_values('contribution', ascending=False)
for _, row in contrib_df.head(5).iterrows():
    print(f"  你喜欢 [{row['liked_title']}] (评分 {row['rating']:.0f})")
    print(f"    -> 与推荐电影相似度 {row['similarity']:.4f}")
    print(f"    -> 贡献分 = {row['rating']:.0f} x {row['similarity']:.4f} = {row['contribution']:.3f}")
    print()

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("总结: ItemCF 协同过滤 — 你刚刚做了什么")
print("=" * 60)
print("""
模块0: 数据采样(3000用户, 7209电影, 42万评分, 稀疏度 1.93%)
模块1: 构建用户-电影矩阵(稀疏, 大部分是 NaN)
模块2: 均值中心化(去掉用户打分偏置)
模块3: 余弦相似度(电影的评分模式有多像)
模块4: 共同评分过滤(少于 3 个共同用户的不算)
模块5: 加权推荐(相似度 x 评分 -> 排序)
模块6: 可解释性(追溯为什么推荐 A)

整个过程没有:
  x 损失函数
  x 梯度下降
  x 模型训练
  x 超参数调优
  x 任何 ML 库

只有:
  o 构建矩阵
  o 算相似度
  o 加权求和
  o 排序

这就是推荐系统的第一块砖。
往上走一步是矩阵分解(SVD/ALS), 再往上是你做过的 LightGBM + GNN。

"站在地基上往上看, 才知道每一层解决了什么问题。"
""")
