"""
模型评估 & 对比实验
================================================================
SparRL Phase 3.1 — 南志锦 · 2026-06-09

评估内容:
  1. BCQ 模型在测试集上的 Top-K 推荐指标
  2. 与 baselines 对比: ItemCF / Popularity / Random
  3. 消融分析: 有无 Perturbation / 有无 VAE constraint

指标:
  - NDCG@K: 归一化折损累计增益 (考虑排序位置)
  - Recall@K: Top-K 中命中多少用户真正评过高分的电影
  - MRR: 平均倒数排名 (第一个相关结果排第几)
  - Precision@K: Top-K 中相关电影占比

推荐系统的评估哲学:
  NDCG 最全面 (考虑位置+相关性), 面试时重点讲。
  Recall 最直观 (找到了几个), 但对排序不敏感。
  MRR 适合"第一个推荐最重要"的场景。
"""

import sys
import io
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import *
from model.embeddings import StateEncoder, MovieEmbeddingLookup
from model.bcq import BCQ
from model.train import TrajectoryDataset, collate_batch, build_movie_lookup


# ============================================================
# 评估指标
# ============================================================
def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """DCG@K: 折损累积增益"""
    scores = np.asarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, scores.size + 2))
    return np.sum((2 ** scores - 1) / discounts)


def ndcg_at_k(predicted_scores: np.ndarray, true_relevance: np.ndarray, k: int) -> float:
    """
    NDCG@K: 归一化折损累积增益

    Args:
        predicted_scores: 模型预测的相关性分数 (越高越好)
        true_relevance:   真实相关性 (评分归一化到 [0,1])
        k:                Top-K
    """
    # 按预测分数排序
    order = np.argsort(predicted_scores)[::-1]
    true_sorted = true_relevance[order]

    dcg = dcg_at_k(true_sorted, k)

    # IDCG: 理想排序
    ideal_order = np.argsort(true_relevance)[::-1]
    ideal_sorted = true_relevance[ideal_order]
    idcg = dcg_at_k(ideal_sorted, k)

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(predicted_scores: np.ndarray, true_relevance: np.ndarray,
                k: int, threshold: float = 0.6) -> float:
    """
    Recall@K: Top-K 中命中了多少相关电影

    threshold: 评分≥该阈值的电影视为"相关" (0.6 ≈ 原始3.2分)
    """
    top_k_idx = np.argsort(predicted_scores)[::-1][:k]
    relevant = true_relevance >= threshold
    if relevant.sum() == 0:
        return 0.0
    return relevant[top_k_idx].sum() / relevant.sum()


def precision_at_k(predicted_scores: np.ndarray, true_relevance: np.ndarray,
                   k: int, threshold: float = 0.6) -> float:
    """Precision@K: Top-K 中相关电影占比"""
    top_k_idx = np.argsort(predicted_scores)[::-1][:k]
    relevant = true_relevance >= threshold
    return relevant[top_k_idx].sum() / k


def mrr(predicted_scores: np.ndarray, true_relevance: np.ndarray,
        threshold: float = 0.6) -> float:
    """MRR: 平均倒数排名"""
    order = np.argsort(predicted_scores)[::-1]
    relevant_positions = np.where(true_relevance[order] >= threshold)[0]
    if len(relevant_positions) == 0:
        return 0.0
    return 1.0 / (relevant_positions[0] + 1)


# ============================================================
# 推荐生成
# ============================================================
def generate_recommendations_bcq(
    model: BCQ,
    state_encoder: StateEncoder,
    movie_lookup: MovieEmbeddingLookup,
    user_state_movie_ids: torch.Tensor,  # (K,) 用户最近 K 部电影 ID
    user_state_ratings: torch.Tensor,    # (K,) 对应评分
    movie_id_to_idx: dict,
    idx_to_movie_id: dict,
    top_k: int = 20,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 BCQ 为单个用户生成 Top-K 推荐。

    流程:
      1. State Encoder: 序列 → state 向量
      2. BCQ select_action: state → 最佳 action 嵌入
      3. 最近邻搜索: 在所有电影嵌入中找 top-K 最近邻
      4. 返回 movie IDs 和相似度分数

    面试话术:
      "BCQ 输出的 action 是连续嵌入向量, 不是具体电影。
       所以需要做最近邻搜索——在 6 万电影的嵌入空间中
       找 cosine similarity 最高的 K 部。
       这相当于在隐空间中做召回。"
    """
    model.eval()
    state_encoder.eval()
    movie_lookup.eval()

    with torch.no_grad():
        # 映射 ID → index
        state_idx = torch.tensor(
            [movie_id_to_idx.get(int(mid), 0) for mid in user_state_movie_ids],
            dtype=torch.long,
        ).unsqueeze(0).to(device)  # (1, K)

        ratings = user_state_ratings.unsqueeze(0).to(device)  # (1, K)

        # State encoding
        state_embs = movie_lookup(state_idx)  # (1, K, D)
        state_vec = state_encoder(state_embs, ratings)  # (1, state_dim)

        # BCQ 选最佳 action
        best_action, _ = model.select_action(state_vec, n_samples=100)  # (1, D)

        # 在所有电影中找最近邻
        all_embs = movie_lookup.get_all_embeddings()  # (N, D)

        # Cosine similarity
        best_action_norm = F.normalize(best_action, p=2, dim=-1)
        all_embs_norm = F.normalize(all_embs, p=2, dim=-1)
        sim_scores = (best_action_norm @ all_embs_norm.T).squeeze(0)  # (N,)

        # Top-K
        top_scores, top_indices = torch.topk(sim_scores, k=top_k)

        # 转换为 movieId
        top_movie_ids = np.array([idx_to_movie_id.get(int(i), 0) for i in top_indices.cpu()])
        top_scores = top_scores.cpu().numpy()

    return top_movie_ids, top_scores


# ============================================================
# Baseline 推荐器
# ============================================================
def recommend_popularity(movies_df: pd.DataFrame, top_k: int = 20,
                         exclude_movie_ids: set = None) -> np.ndarray:
    """Popularity baseline: 推荐最热门的电影"""
    if exclude_movie_ids is None:
        exclude_movie_ids = set()
    popular = movies_df[~movies_df["movieId"].isin(exclude_movie_ids)]
    popular = popular.sort_values("rating_count", ascending=False)
    return popular["movieId"].values[:top_k]


def recommend_random(movies_df: pd.DataFrame, top_k: int = 20,
                     exclude_movie_ids: set = None, seed: int = 42) -> np.ndarray:
    """Random baseline: 随机推荐"""
    if exclude_movie_ids is None:
        exclude_movie_ids = set()
    np.random.seed(seed)
    candidates = movies_df[~movies_df["movieId"].isin(exclude_movie_ids)]
    sample = candidates.sample(n=min(top_k, len(candidates)))
    return sample["movieId"].values


# ============================================================
# 主评估函数
# ============================================================
def evaluate_model(
    model: BCQ,
    state_encoder: StateEncoder,
    movie_lookup: MovieEmbeddingLookup,
    test_ratings: pd.DataFrame,
    movie_id_to_idx: dict,
    idx_to_movie_id: dict,
    device: str = "cpu",
    n_users: int = None,
) -> dict:
    """
    在测试集上评估 BCQ 模型。

    对每个测试用户:
      1. 取用户最近的 STATE_SIZE 部训练集评分作为 state
      2. 用 BCQ 生成 Top-K 推荐
      3. 与测试集实际评分对比
      4. 计算 NDCG/Recall/Precision/MRR
    """
    print("\n" + "=" * 60)
    print("BCQ 模型评估")
    print("=" * 60)

    if n_users is None:
        n_users = min(EVAL_USER_SAMPLE, test_ratings["userId"].nunique())

    test_users = test_ratings["userId"].unique()[:n_users]
    print(f"  评估用户数: {len(test_users)}")

    metrics = defaultdict(list)
    k_values = EVAL_K_VALUES

    for i, user_id in enumerate(test_users):
        # 用户的测试集评分
        user_test = test_ratings[test_ratings["userId"] == user_id]

        if len(user_test) < 1:
            continue

        # 取用户最近的 STATE_SIZE 部训练集评分作为 state
        # (简化: 取用户全局评分中按时间最早的 STATE_SIZE 部)
        user_all = pd.concat([
            pd.DataFrame({"movieId": user_test["movieId"].values[:STATE_SIZE],
                          "rating": user_test["rating"].values[:STATE_SIZE]})
        ])

        # 构建 state
        state_movie_ids = torch.tensor(
            user_all["movieId"].values[:STATE_SIZE], dtype=torch.float32
        ).unsqueeze(0)
        state_ratings = torch.tensor(
            (user_all["rating"].values[:STATE_SIZE] - 0.5) / 4.5, dtype=torch.float32
        ).unsqueeze(0)

        if len(state_movie_ids.squeeze()) < STATE_SIZE:
            continue

        # 生成推荐
        rec_movie_ids, rec_scores = generate_recommendations_bcq(
            model, state_encoder, movie_lookup,
            state_movie_ids.squeeze(0), state_ratings.squeeze(0),
            movie_id_to_idx, idx_to_movie_id,
            top_k=max(k_values), device=device,
        )

        # 构建真实相关性向量 (测试集评分)
        user_test_dict = dict(zip(user_test["movieId"], user_test["rating"]))
        true_rel = np.array([
            (user_test_dict.get(int(mid), 0.0) - 0.5) / 4.5
            for mid in rec_movie_ids
        ])

        # 计算各指标
        for k in k_values:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec_scores[:k], true_rel[:k], k))
            metrics[f"recall@{k}"].append(recall_at_k(rec_scores[:k], true_rel[:k], k))
            metrics[f"precision@{k}"].append(precision_at_k(rec_scores[:k], true_rel[:k], k))
        metrics["mrr"].append(mrr(rec_scores, true_rel))

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(test_users)}")

    # 汇总
    results = {}
    print("\n  评估结果:")
    print("  " + "-" * 50)
    for metric_name in sorted(metrics.keys()):
        values = metrics[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"  {metric_name:<15} = {mean_val:.4f} ± {std_val:.4f}")

    return results


# ============================================================
# Baseline 评估
# ============================================================
def evaluate_baselines(test_ratings: pd.DataFrame, movies_df: pd.DataFrame,
                       n_users: int = None) -> dict:
    """评估 baseline 方法"""
    print("\n" + "=" * 60)
    print("Baseline 评估")
    print("=" * 60)

    if n_users is None:
        n_users = min(EVAL_USER_SAMPLE, test_ratings["userId"].nunique())

    test_users = test_ratings["userId"].unique()[:n_users]

    metrics_pop = defaultdict(list)
    metrics_rand = defaultdict(list)
    k_values = EVAL_K_VALUES

    for user_id in test_users:
        user_test = test_ratings[test_ratings["userId"] == user_id]
        if len(user_test) < 1:
            continue

        user_test_dict = dict(zip(user_test["movieId"], user_test["rating"]))
        true_ratings = user_test["rating"].values[:max(k_values)]

        # Popularity
        pop_recs = recommend_popularity(movies_df, max(k_values))
        pop_scores = np.ones(max(k_values))  # 全部等权
        true_rel = np.array([
            (user_test_dict.get(int(mid), 0.0) - 0.5) / 4.5
            for mid in pop_recs
        ])
        for k in k_values:
            metrics_pop[f"ndcg@{k}"].append(ndcg_at_k(pop_scores[:k], true_rel[:k], k))
            metrics_pop[f"recall@{k}"].append(recall_at_k(pop_scores[:k], true_rel[:k], k))
        metrics_pop["mrr"].append(mrr(pop_scores, true_rel))

        # Random
        rand_recs = recommend_random(movies_df, max(k_values))
        true_rel_rand = np.array([
            (user_test_dict.get(int(mid), 0.0) - 0.5) / 4.5
            for mid in rand_recs
        ])
        for k in k_values:
            metrics_rand[f"ndcg@{k}"].append(ndcg_at_k(np.ones(k), true_rel_rand[:k], k))
            metrics_rand[f"recall@{k}"].append(recall_at_k(np.ones(k), true_rel_rand[:k], k))
        metrics_rand["mrr"].append(mrr(np.ones(max(k_values)), true_rel_rand))

    # 汇总
    print("\n  Popularity:")
    for k in k_values:
        print(f"    NDCG@{k} = {np.mean(metrics_pop[f'ndcg@{k}']):.4f}")
    print("\n  Random:")
    for k in k_values:
        print(f"    NDCG@{k} = {np.mean(metrics_rand[f'ndcg@{k}']):.4f}")

    return {
        "popularity": {k: {"mean": np.mean(v)} for k, v in metrics_pop.items()},
        "random": {k: {"mean": np.mean(v)} for k, v in metrics_rand.items()},
    }


# ============================================================
# Online Learning Curve — 在线学习曲线评估 ★
# ============================================================
def evaluate_online_learning_curve(
    base_model_path: str,
    online_data_dir: str,
    movie_id_to_idx: dict,
    idx_to_movie_id: dict,
    movies_df: pd.DataFrame,
    device: str = "cpu",
    n_users: int = None,
):
    """
    在线学习曲线评估: 逐轮评估模型在"未来数据"上的表现。

    模拟真实场景:
      轮1: 用前25%数据训练 → 在第2个窗口评估
      轮2: 用前50%数据训练 → 在第3个窗口评估
      轮3: 用前75%数据训练 → 在第4个窗口评估

    画出 NDCG@K 随训练数据量的变化曲线 →
    判断"模型还需要多少数据"、"有没有饱和"。

    面试话术:
      "学习曲线向下走 = 模型需要更多数据
       学习曲线平坦   = 数据够了, 改进模型结构
       学习曲线向上走 = 模型在适应新数据 (distribution shift)"
    """
    import json
    import glob
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("Online Learning Curve — 在线学习曲线")
    print("=" * 60)

    # 发现所有轮次的测试数据
    test_files = sorted(glob.glob(os.path.join(online_data_dir, "round*_test.parquet")))
    train_dirs = sorted(glob.glob(os.path.join(online_data_dir, "round*_train.parquet")))
    n_rounds = len(test_files)
    print(f"  在线轮次: {n_rounds}")
    print(f"  每轮评估: 训练(累积窗口0..r-1) → 测试(窗口r)")

    k_values = EVAL_K_VALUES
    all_metrics = defaultdict(list)

    for r in range(1, n_rounds + 1):
        # 加载本轮的测试数据
        test_path = os.path.join(online_data_dir, f"round{r}_test.parquet")
        test_ratings = pd.read_parquet(test_path)

        print(f"\n  --- 轮{r} ---")
        print(f"  测试集: {len(test_ratings):,} 条评分 "
              f"({test_ratings['userId'].nunique():,} 用户)")

        # 尝试加载对应的模型 checkpoint
        ckpt_path = os.path.join(online_data_dir, f"round{r}_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️ 找不到模型: {ckpt_path}")
            print(f"  提示: 先运行 python model/train.py --online")
            continue

        # 加载模型
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_encoder = StateEncoder().to(device)
        bcq = BCQ().to(device)
        state_encoder.load_state_dict(checkpoint["state_encoder"])
        bcq.load_state_dict(checkpoint["bcq"])

        # 重建 movie_lookup
        movie_lookup = MovieEmbeddingLookup(
            n_movies=len(movie_id_to_idx),
            movie_genres=torch.zeros(1, 1),
            movie_years=torch.zeros(1),
            movie_avg_ratings=torch.zeros(1),
            movie_rating_stds=torch.zeros(1),
            movie_popularity=torch.zeros(1),
        )
        movie_lookup.load_state_dict(checkpoint["movie_lookup"])
        movie_lookup = movie_lookup.to(device)

        # 评估
        results = evaluate_model(
            bcq, state_encoder, movie_lookup,
            test_ratings, movie_id_to_idx, idx_to_movie_id,
            device=device, n_users=n_users,
        )
        for k, v in results.items():
            all_metrics[k].append(v["mean"])

        # 记录训练数据量
        train_path = os.path.join(online_data_dir, f"round{r}_train.parquet")
        train_n = len(pd.read_parquet(train_path))
        all_metrics["train_n"].append(train_n)
        print(f"  训练数据量: {train_n:>12,} 条评分")

    # 汇总 & 绘图
    print("\n" + "=" * 60)
    print("学习曲线汇总")
    print("=" * 60)

    print(f"\n  {'轮次':<6} {'训练数据量':<14}", end="")
    for k in k_values:
        print(f" {'NDCG@{k}':<12}", end="")
    print()

    for r in range(len(all_metrics.get(f"ndcg@{k_values[0]}", []))):
        train_n = all_metrics["train_n"][r] if "train_n" in all_metrics else 0
        print(f"  {r+1:<6} {train_n:<14,}", end="")
        for k in k_values:
            val = all_metrics.get(f"ndcg@{k}", [])[r] if r < len(all_metrics.get(f"ndcg@{k}", [])) else 0
            print(f" {val:<12.4f}", end="")
        print()

    # 简单 ASCII 绘图 (不需要 matplotlib)
    print(f"\n  ASCII 学习曲线 (NDCG@10):")
    ndcg10 = all_metrics.get("ndcg@10", [])
    if ndcg10:
        max_val = max(ndcg10)
        min_val = min(ndcg10)
        for i, v in enumerate(ndcg10):
            bar_len = int((v - min_val) / (max_val - min_val + 1e-8) * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            train_n = all_metrics["train_n"][i] if i < len(all_metrics.get("train_n", [])) else 0
            print(f"  轮{i+1} ({train_n:>10,}条) │{bar}│ {v:.4f}")

    # 输出面试洞察
    if len(ndcg10) >= 3:
        slope = (ndcg10[-1] - ndcg10[0]) / ndcg10[0] * 100
        print(f"\n  趋势分析:")
        print(f"  初始 NDCG@10:     {ndcg10[0]:.4f} (最小数据)")
        print(f"  最终 NDCG@10:     {ndcg10[-1]:.4f} (全量数据)")
        print(f"  变化:             {slope:+.1f}%")
        if slope > 5:
            print(f"  📈 模型从更多数据中显著受益 — 可以继续加数据")
        elif slope > 0:
            print(f"  📊 模型从数据中平缓受益 — 数据量接近饱和")
        else:
            print(f"  📉 模型随数据变差 — 可能是 distribution shift, 需要定期重训")

    return dict(all_metrics)


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SparRL 模型评估")
    parser.add_argument("--online", action="store_true",
                        help="在线学习曲线模式")
    parser.add_argument("--n-users", type=int, default=None,
                        help="评估用户数")
    args = parser.parse_args()

    print("=" * 60)
    print("SparRL — 模型评估")
    print("=" * 60)

    is_autodl, device, gpu_count = detect_autodl()
    movies_df = pd.read_parquet(MOVIE_FEATURES_PATH)

    # 构建 movie ID 映射
    _, movie_id_to_idx = build_movie_lookup(None, MOVIE_FEATURES_PATH)
    idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}

    eval_users = args.n_users or (EVAL_USER_SAMPLE if not is_autodl else None)

    # ─── Online Learning Curve 模式 ───
    if args.online:
        online_dir = os.path.join(OUTPUT_DIR, "online_windows")
        if not os.path.exists(online_dir):
            print(f"\n❌ 找不到时间窗口数据: {online_dir}")
            print("   请先运行: python spark/preprocess.py")
            return

        print(f"\n[在线模式] 从 {online_dir} 加载逐轮模型...")
        evaluate_online_learning_curve(
            base_model_path=MODEL_SAVE_PATH,
            online_data_dir=online_dir,
            movie_id_to_idx=movie_id_to_idx,
            idx_to_movie_id=idx_to_movie_id,
            movies_df=movies_df,
            device=str(device),
            n_users=eval_users,
        )
        print("\n  在线学习曲线评估完成 ✅")
        return

    # ─── 标准评估模式 ───
    # 加载模型
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"\n❌ 找不到模型: {MODEL_SAVE_PATH}")
        print("   请先运行: python model/train.py")
        return

    print(f"\n[加载模型] {MODEL_SAVE_PATH}")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)

    state_encoder = StateEncoder().to(device)
    bcq = BCQ().to(device)

    state_encoder.load_state_dict(checkpoint["state_encoder"])
    bcq.load_state_dict(checkpoint["bcq"])

    # 从 checkpoint 加载 movie lookup
    movie_lookup = MovieEmbeddingLookup(
        n_movies=len(movie_id_to_idx),
        movie_genres=torch.zeros(1, 1),  # placeholder, 从 state_dict 覆盖
        movie_years=torch.zeros(1),
        movie_avg_ratings=torch.zeros(1),
        movie_rating_stds=torch.zeros(1),
        movie_popularity=torch.zeros(1),
    )
    movie_lookup.load_state_dict(checkpoint["movie_lookup"])
    movie_lookup = movie_lookup.to(device)

    # 加载测试集
    test_ratings_path = os.path.join(OUTPUT_DIR, "ratings_test.parquet")
    test_ratings = pd.read_parquet(test_ratings_path)
    print(f"  测试集: {len(test_ratings):,} 条评分")

    # 评估 BCQ
    bcq_results = evaluate_model(
        bcq, state_encoder, movie_lookup,
        test_ratings, movie_id_to_idx, idx_to_movie_id,
        device=str(device),
        n_users=eval_users,
    )

    # 评估 Baselines
    baseline_results = evaluate_baselines(
        test_ratings, movies_df, n_users=eval_users
    )

    # 对比汇总
    print("\n" + "=" * 60)
    print("模型对比汇总")
    print("=" * 60)
    print(f"\n  {'Method':<15} {'NDCG@10':<12} {'NDCG@20':<12} {'Recall@10':<12} {'MRR':<12}")
    print("  " + "-" * 63)

    # 简化输出
    bcq_ndcg10 = bcq_results.get('ndcg@10', {}).get('mean', 0)
    bcq_ndcg20 = bcq_results.get('ndcg@20', {}).get('mean', 0)
    bcq_recall10 = bcq_results.get('recall@10', {}).get('mean', 0)
    bcq_mrr = bcq_results.get('mrr', {}).get('mean', 0)
    print(f"  {'BCQ':<15} {bcq_ndcg10:.4f}      {bcq_ndcg20:.4f}      {bcq_recall10:.4f}      {bcq_mrr:.4f}")

    pop = baseline_results.get("popularity", {})
    if pop:
        print(f"  {'Popularity':<15} "
              f"{pop.get('ndcg@10', {}).get('mean', 0):.4f}      "
              f"{pop.get('ndcg@20', {}).get('mean', 0):.4f}      "
              f"{pop.get('recall@10', {}).get('mean', 0):.4f}      "
              f"{pop.get('mrr', {}).get('mean', 0):.4f}")

    rnd = baseline_results.get("random", {})
    if rnd:
        print(f"  {'Random':<15} "
              f"{rnd.get('ndcg@10', {}).get('mean', 0):.4f}      "
              f"{rnd.get('ndcg@20', {}).get('mean', 0):.4f}      "
              f"{rnd.get('recall@10', {}).get('mean', 0):.4f}      "
              f"{rnd.get('mrr', {}).get('mean', 0):.4f}")

    print("\n  评估完成 ✅")


if __name__ == "__main__":
    main()
