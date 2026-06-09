"""
Offline Policy Evaluation (OPE) — Spark 分布式评估
================================================================
SparRL Phase 3.2 — 南志锦 · 2026-06-09

OPE (离线策略评估) 是 offline RL 的核心技术之一。
不用上线就能估计一个新策略的效果——面试时这是很好的区分点。

三种经典估计器 (在 Spark 上分布式计算):

  1. IPS (Inverse Propensity Scoring)
     思路: 用重要性采样加权, 纠正"行为策略 vs 目标策略"的分布差异
     公式: V_IPS = (1/N) Σ w_i · r_i
          其中 w_i = π_e(a_i|s_i) / π_b(a_i|s_i)
     优点: 无偏
     缺点: 方差大 (权重可能很大)

  2. DM (Direct Method)
     思路: 用学习到的 reward model 直接估计
     公式: V_DM = (1/N) Σ r̂(s_i, π_e(s_i))
     优点: 低方差
     缺点: 有偏 (依赖 reward model 准确性)

  3. DR (Doubly Robust)
     思路: IPS + DM 结合, 双保险
     公式: V_DR = V_DM + (1/N) Σ w_i · (r_i - r̂(s_i, a_i))
     优点: IPS 无偏 + DM 低方差
     面试金句: "只要 reward model 或 propensity model 有一个对, DR 就是无偏的"

Spark 角色:
  这三类估计器都涉及"对每个 (state, action) 计算一个值然后平均"——
  天然的 embarrassingly parallel 计算。Spark 可以:
    1. 广播模型权重到所有 executor
    2. 每个 executor 计算自己 partition 的贡献
    3. collect 汇总

面试话术:
  "OPE 的价值在于: 推荐系统上线前可以离线评估策略效果。
   不需要 A/B 测试就能排除掉差策略, 降低实验成本。
   IPS 的方差问题在推荐场景特别严重——因为 item 空间巨大,
   很多 action 的 propensity 极低, 导致权重爆炸。
   实际工业界常用 clipping 或 self-normalization 缓解。"
"""

import sys
import io
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import *


# ============================================================
# 模块0 — Spark 环境
# ============================================================
def init_spark() -> SparkSession:
    builder = SparkSession.builder
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, str(value))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# 模块1 — 加载测试轨迹
# ============================================================
def load_test_trajectories(spark: SparkSession) -> tuple:
    """
    加载测试集轨迹数据用于 OPE。

    注意: OPE 评估的是"在行为策略收集的数据上,
    评估目标策略的表现", 所以用的是测试集 (行为策略) 的数据,
    但用目标策略 (BCQ) 来做 counterfactual 判断。
    """
    print("\n[模块1] 加载测试集评分...")

    test_ratings_path = os.path.join(OUTPUT_DIR, "ratings_test.parquet")
    test_ratings = spark.read.parquet(test_ratings_path)
    print(f"  测试评分: {test_ratings.count():,} 行")
    return test_ratings


# ============================================================
# 模块2 — IPS 估计器
# ============================================================
def compute_ips(
    spark: SparkSession,
    test_ratings,
    propensity_model=None,
    clip_threshold: float = 10.0,
):
    """
    IPS (Inverse Propensity Scoring) 估计器

    在 Spark 上分布式计算:
      V_IPS = (1/|D|) · Σ min(w_i, C) · r_i

    其中:
      w_i = π_e(a_i|s_i) / π_b(a_i|s_i)  (importance weight)
      C = clip_threshold (防止方差爆炸)

    面试话术:
      "IPS 的核心假设是 SUTVA (Stable Unit Treatment Value Assumption):
       一个用户的结果不受其他用户被推荐什么的影响。
       在推荐系统里这个假设通常成立, 因为推荐是个性化的。

       propensity clipping 是工业界标配——
       把权重截断到 [1/C, C], 牺牲一点无偏性换取稳定性。"
    """
    print("\n[模块2] IPS 估计器")

    # 简化: 用评分频率作为行为策略 propensity 的代理
    # π_b(a|s) ≈ count(a) / total_ratings
    total_count = test_ratings.count()

    # 每部电影的 propensity (被评概率)
    movie_propensity = test_ratings.groupBy("movieId").agg(
        (F.count("*") / total_count).alias("propensity")
    )

    # 关联到测试评分
    test_with_prop = test_ratings.join(
        F.broadcast(movie_propensity), on="movieId"
    )

    # 简化 IPS: 对于 reward, 按照 1/propensity 加权
    # (完全精确的 IPS 还需要 target policy 的概率, 这里用
    #  uniform target policy 做演示 — 实际使用时替换为 BCQ 策略)
    ips_result = test_with_prop.agg(
        F.avg(F.col("rating") / F.col("propensity")).alias("ips_value"),
        F.stddev(F.col("rating") / F.col("propensity")).alias("ips_std"),
        F.avg(F.when(
            F.col("rating") / F.col("propensity") > clip_threshold,
            clip_threshold
        ).otherwise(
            F.col("rating") / F.col("propensity")
        )).alias("ips_clipped"),
    ).collect()[0]

    # Self-normalized IPS (更稳定)
    weights = test_with_prop.select(1.0 / F.col("propensity"))
    sum_w = weights.agg(F.sum("value")).collect()[0][0]
    sn_ips = test_with_prop.agg(
        F.sum(F.col("rating") / F.col("propensity")).alias("weighted_sum")
    ).collect()[0]["weighted_sum"] / sum_w if sum_w > 0 else 0.0

    print(f"  IPS (raw):           {ips_result['ips_value']:.4f} ± {ips_result['ips_std']:.4f}")
    print(f"  IPS (clipped@{clip_threshold}):    {ips_result['ips_clipped']:.4f}")
    print(f"  IPS (self-normalized): {sn_ips:.4f}")

    return {
        "ips_raw": ips_result["ips_value"],
        "ips_std": ips_result["ips_std"],
        "ips_clipped": ips_result["ips_clipped"],
        "ips_sn": sn_ips,
    }


# ============================================================
# 模块3 — Direct Method 估计器
# ============================================================
def compute_dm(spark, test_ratings):
    """
    DM (Direct Method) 估计器

    思路: 不依赖 propensity, 直接用 reward model 预测。
    V_DM = (1/N) Σ r̂(s_i, a_i)

    这里用全局平均评分作为最简单的 reward model (baseline)。
    实际使用时替换为 BCQ 的 Q 网络预测值。

    面试话术:
      "DM 不依赖 importance sampling, 所以方差低。
       但它完全依赖 reward model 的准确性——
       如果模型对某些 (state, action) 的预测偏差很大,
       DM 的估计也会偏。这就是为什么 DR 要结合两者。"
    """
    print("\n[模块3] Direct Method 估计器")

    # 简化 reward model: 每部电影的全局平均分
    movie_avg_rating = test_ratings.groupBy("movieId").agg(
        F.avg("rating").alias("predicted_reward")
    )

    test_with_pred = test_ratings.join(
        F.broadcast(movie_avg_rating), on="movieId"
    )

    dm_result = test_with_pred.agg(
        F.avg("predicted_reward").alias("dm_value"),
        F.avg(F.col("rating") - F.col("predicted_reward")).alias("dm_bias"),
    ).collect()[0]

    print(f"  DM value: {dm_result['dm_value']:.4f}")
    print(f"  DM bias:  {dm_result['dm_bias']:.4f} "
          f"(<0 表示 reward model 低估, >0 表示高估)")

    return {
        "dm_value": dm_result["dm_value"],
        "dm_bias": dm_result["dm_bias"],
    }


# ============================================================
# 模块4 — Doubly Robust 估计器
# ============================================================
def compute_dr(spark, test_ratings, clip_threshold: float = 10.0):
    """
    DR (Doubly Robust) 估计器

    V_DR = V_DM + (1/N) Σ w_i · (r_i - r̂(s_i, a_i))

    其中 V_DM 和 r̂ 来自 Direct Method,
    w_i 来自 IPS 的重要性权重。

    面试话术:
      "DR 的精妙之处在于双重鲁棒性:
       - 如果 propensity model 对, 第二项给出无偏修正 → 整体无偏
       - 如果 reward model 对, r_i - r̂ = 0 → V_DR = V_DM → 无偏
       - 两者都对 → DR 既无偏又低方差
       实际中很难两个 model 都完美, 但只要有一个够好就行。"
    """
    print("\n[模块4] Doubly Robust 估计器")

    total_count = test_ratings.count()

    # Propensity model (same as IPS)
    movie_propensity = test_ratings.groupBy("movieId").agg(
        (F.count("*") / total_count).alias("propensity")
    )

    # Reward model (same as DM)
    movie_avg_rating = test_ratings.groupBy("movieId").agg(
        F.avg("rating").alias("predicted_reward")
    )

    test_full = test_ratings.join(
        F.broadcast(movie_propensity), on="movieId"
    ).join(
        F.broadcast(movie_avg_rating), on="movieId"
    )

    # DR 计算
    dr_result = test_full.agg(
        # V_DM: 对所有可能的 action 取预测 reward 的均值
        F.avg("predicted_reward").alias("v_dm"),
        # 修正项: w_i · (r_i - r̂_i)
        F.avg(
            (1.0 / F.col("propensity")) *
            (F.col("rating") - F.col("predicted_reward"))
        ).alias("correction_term"),
        # Clipped version
        F.avg(
            F.when(
                1.0 / F.col("propensity") > clip_threshold,
                clip_threshold
            ).otherwise(1.0 / F.col("propensity")) *
            (F.col("rating") - F.col("predicted_reward"))
        ).alias("correction_clipped"),
    ).collect()[0]

    v_dr = dr_result["v_dm"] + dr_result["correction_term"]
    v_dr_clipped = dr_result["v_dm"] + dr_result["correction_clipped"]
    actual_avg = test_full.agg(F.avg("rating")).collect()[0][0]

    print(f"  V_DM:           {dr_result['v_dm']:.4f}")
    print(f"  修正项 (IPS):    {dr_result['correction_term']:.4f}")
    print(f"  V_DR:           {v_dr:.4f}")
    print(f"  V_DR (clipped): {v_dr_clipped:.4f}")
    print(f"  实际平均评分:    {actual_avg:.4f}")

    return {
        "v_dm": dr_result["v_dm"],
        "correction_term": dr_result["correction_term"],
        "v_dr": v_dr,
        "v_dr_clipped": v_dr_clipped,
        "actual_avg": actual_avg,
    }


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print("SparRL — OPE 离线策略评估")
    print("=" * 60)

    spark = init_spark()

    try:
        test_ratings = load_test_trajectories(spark)

        print(f"\n{'='*60}")
        print("OPE 三件套")
        print(f"{'='*60}")
        print("""
  面试记忆口诀:
    IPS  → 重加权, 无偏但方差大
    DM   → 直接估, 低方差但有偏
    DR   → 双保险, 对一半就无偏
""")

        results_ips = compute_ips(spark, test_ratings)
        results_dm = compute_dm(spark, test_ratings)
        results_dr = compute_dr(spark, test_ratings)

        # 汇总
        print(f"\n{'='*60}")
        print("OPE 汇总")
        print(f"{'='*60}")
        print(f"""
  {"估计器":<20} {"值":<10} {"特点":<30}
  {"-"*60}
  {"IPS (raw)":<20} {results_ips['ips_raw']:.4f}     {"无偏, 高方差":<30}
  {"IPS (clipped)":<20} {results_ips['ips_clipped']:.4f}     {"有偏, 低方差":<30}
  {"IPS (SN)":<20} {results_ips['ips_sn']:.4f}     {"Self-Normalized":<30}
  {"DM":<20} {results_dm['dm_value']:.4f}     {"完全依赖 reward model":<30}
  {"DR":<20} {results_dr['v_dr']:.4f}     {"双保险 (推荐使用)":<30}
  {"DR (clipped)":<20} {results_dr['v_dr_clipped']:.4f}     {"DR + 方差控制":<30}
  {"实际均值":<20} {results_dr['actual_avg']:.4f}     {"测试集真实平均评分":<30}
""")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
