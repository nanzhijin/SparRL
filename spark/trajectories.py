"""
Spark RL 轨迹构建
================================================================
SparRL Phase 1.3 — 南志锦 · 2026-06-09

这是整个项目最核心的 Spark 代码——用窗口函数把
2500万条静态评分记录变成 500万条 RL 训练轨迹。

核心概念:
  传统推荐: (user, movie) → rating  (监督学习)
  RL 推荐:   (state, action) → reward → next_state  (序列决策)

  state  = 用户最近 K 部的评分历史  (上下文)
  action = 推荐哪部电影            (决策)
  reward = 评分                    (反馈)
  next_state = 看过这部电影后的新历史

Spark 窗口函数是完成这个"范式转换"的关键工具。

面试核心考点:
  ⭐ PARTITION BY userId ORDER BY timestamp → 窗口函数
  ⭐ ROWS BETWEEN K PRECEDING AND 1 PRECEDING → 滑动窗口
  ⭐ collect_list → 聚合窗口内的数据为 Array
  ⭐ 宽依赖 vs 窄依赖 → PARTITION BY 不触发额外 Shuffle
  ⭐ 数据倾斜 → 大用户(评分多)可能导致分区不均
"""

import sys
import io
import os
import time
from pyspark.sql import SparkSession, DataFrame
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
    print(f"[Spark] {spark.version} | {spark.sparkContext.master} | "
          f"cores={spark.sparkContext.defaultParallelism}")
    return spark


# ============================================================
# 模块1 — 加载训练集评分
# ============================================================
def load_train_ratings(spark: SparkSession) -> DataFrame:
    """
    加载 preprocess.py 输出的训练集评分。

    面试话术:
      parquet 是列式存储, 压缩率通常 5-10x, 自带 schema。
      Spark 读取 parquet 时可以谓词下推 (predicate pushdown):
      如果后面有 filter, Spark 会在扫描文件时就跳过不匹配的行块。
    """
    ratings_train_path = os.path.join(OUTPUT_DIR, "ratings_train.parquet")

    ratings = spark.read.parquet(ratings_train_path)
    n = ratings.count()
    n_users = ratings.select("userId").distinct().count()
    print(f"\n[模块1] 训练集评分: {n:,} 行, {n_users:,} 用户")
    return ratings


# ============================================================
# 模块2 — 构建轨迹（核心 Spark 逻辑）
# ============================================================
def build_trajectories(ratings: DataFrame) -> DataFrame:
    """
    用 Spark 窗口函数把评分序列变成 RL 轨迹。

    ★ 这是整个项目的 Spark 技术制高点 ★

    算法:
      1. 每个用户按时间排序 → ROW_NUMBER
      2. 滑窗 collect_list 获得 state 和 next_state
      3. 当前行的 movieId = action, rating = reward
      4. 过滤掉 state 不完整的行 (前 K 条评分只能当 state, 不能做样本)

    SQL 等价写法 (面试可画):
      WITH ordered AS (
        SELECT userId, movieId, rating, timestamp,
          ROW_NUMBER() OVER (
            PARTITION BY userId ORDER BY timestamp
          ) AS rn
        FROM ratings_train
      ),
      -- 核心: 窗口聚合构建 state
      with_state AS (
        SELECT *,
          COLLECT_LIST(STRUCT(movieId, rating)) OVER (
            PARTITION BY userId ORDER BY rn
            ROWS BETWEEN {K} PRECEDING AND 1 PRECEDING
          ) AS state,
          COLLECT_LIST(STRUCT(movieId, rating)) OVER (
            PARTITION BY userId ORDER BY rn
            ROWS BETWEEN {K-1} PRECEDING AND CURRENT ROW
          ) AS next_state
        FROM ordered
      )
      SELECT userId, rn, state, movieId AS action, rating AS reward, next_state
      FROM with_state
      WHERE rn > {K}

    面试话术 — 窗口函数原理:
      "PARTITION BY userId 把数据按用户分组, ORDER BY timestamp 组内排序。
       关键认知: PARTITION BY 不触发 Shuffle!
       因为数据在上游 groupBy 用户统计特征时已经按 userId 分好区了,
       Spark 优化器会识别出 '数据已按 key 分布', 跳过 Shuffle 步骤。
       这就是 AQE (Adaptive Query Execution) 的威力。"

    面试话术 — 滑窗 ROWS BETWEEN:
      "ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING 是物理行窗口,
       不是 RANGE (值范围)。ROWS 模式下每个分区的缓存只需
       保留最近 K+1 行, 内存开销 O(K), 与用户评分数无关。
       这就是为什么即使用户评了 10 万部电影, 这个查询也不会 OOM。"

    面试话术 — collect_list:
      "collect_list 是聚合函数, 在窗口内把多行数据收集成一个 Array。
       配合 STRUCT 可以保持多字段关联 (movieId + rating 不拆散)。
       注意: collect_list 在窗口模式下不会去重, 保留窗口内的所有行。
       如果数据量大, 可以用 collect_set 去重或增加内存分区控制。"
    """
    print("\n" + "=" * 60)
    print("模块2 — 构建 RL 轨迹 (Spark 窗口函数)")
    print("=" * 60)

    K = STATE_SIZE
    t0 = time.time()

    # Step 1: 每个用户内按时间排序, 分配行号
    # ──────────────────────────────────────────
    # row_number() OVER (PARTITION BY userId ORDER BY timestamp)
    window_order = Window.partitionBy("userId").orderBy("timestamp")

    ordered = ratings.withColumn("rn", F.row_number().over(window_order))

    # 同时计算该用户的总评分数 (用于后续统计)
    ordered = ordered.withColumn(
        "total_ratings",
        F.max("rn").over(Window.partitionBy("userId")),
    )

    # Step 2: 滑窗构建 state 和 next_state
    # ─────────────────────────────────────
    # state: 当前行之前的 K 条 (不含当前行)
    #   ROWS BETWEEN K PRECEDING AND 1 PRECEDING
    state_window = window_order.rowsBetween(-K, -1)

    # next_state: 当前行 + 之前的 K-1 条
    #   ROWS BETWEEN (K-1) PRECEDING AND CURRENT ROW
    next_state_window = window_order.rowsBetween(-(K - 1), 0)

    # 用 struct 保持 (movieId, rating) 关联不散
    ordered = ordered.withColumn(
        "state_structs",
        F.collect_list(F.struct("movieId", "rating")).over(state_window),
    ).withColumn(
        "next_state_structs",
        F.collect_list(F.struct("movieId", "rating")).over(next_state_window),
    )

    # Step 3: 过滤 + 提取
    # ────────────────────
    # 只有 rn > K 的行才能构成完整训练样本 (前面的 K 行不够 state)
    trajectories = ordered.filter(
        (F.col("rn") > K) &
        (F.size("state_structs") == K)
    )

    # 从 struct array 中拆出独立的 array 列 (方便 PyTorch 读取)
    trajectories = trajectories.select(
        "userId",
        "rn",
        "total_ratings",
        "timestamp",
        # state: 前 K 条
        F.col("state_structs.movieId").alias("state_movie_ids"),
        F.col("state_structs.rating").alias("state_ratings"),
        # action + reward (当前行)
        F.col("movieId").alias("action_movie_id"),
        F.col("rating").alias("reward"),
        # next_state: 包含当前行的最近 K 条
        F.col("next_state_structs.movieId").alias("next_state_movie_ids"),
        F.col("next_state_structs.rating").alias("next_state_ratings"),
    )

    elapsed = time.time() - t0
    n_trajectories = trajectories.count()

    print(f"\n  轨迹构建耗时: {elapsed:.1f}s")
    print(f"  训练样本数:   {n_trajectories:,}")
    print(f"  State 大小:   K={K}")
    print(f"  样本/用户:    {n_trajectories / trajectories.select('userId').distinct().count():.1f}")

    # 快速检查数据质量
    print("\n  轨迹样例 (前3条):")
    trajectories.select(
        "userId", "rn", "action_movie_id", "reward",
        F.size("state_movie_ids").alias("state_len"),
    ).show(3, truncate=False)

    return trajectories


# ============================================================
# 模块3 — 轨迹质量分析
# ============================================================
def analyze_trajectories(trajectories: DataFrame):
    """
    分析构建好的轨迹数据质量。

    面试话术:
      数据质量检查是建模的前置步骤。在 Spark 上做 describe
      可以一次性拿到所有列的分布, 因为 Spark 会并行计算
      各列的 count/mean/stddev/min/max。
    """
    print("\n" + "=" * 60)
    print("模块3 — 轨迹质量分析")
    print("=" * 60)

    # Reward 分布 (评分分布)
    print("\n  Reward 分布:")
    trajectories.select("reward").describe().show()
    trajectories.groupBy("reward").count().orderBy("reward").show()

    # State 长度分布 (应该都等于 STATE_SIZE)
    state_len_dist = trajectories.withColumn(
        "state_len", F.size("state_movie_ids")
    ).groupBy("state_len").count().orderBy("state_len")
    print("\n  State 长度分布:")
    state_len_dist.show()

    # 每个用户样本数分布
    user_sample_count = trajectories.groupBy("userId").count()
    print("\n  每用户轨迹数分布:")
    user_sample_count.select("count").describe().show()

    # 大用户警告 (数据倾斜风险)
    p99 = user_sample_count.stat.approxQuantile("count", [0.99], 0.01)[0]
    max_samples = user_sample_count.agg(F.max("count")).collect()[0][0]
    print(f"\n  99% 分位数: {p99:.0f} 条/用户")
    print(f"  最大值:      {max_samples} 条/用户")
    if max_samples > p99 * 5:
        print(f"  ⚠️ 数据倾斜警告: 最大值是 p99 的 {max_samples/p99:.1f}x")
        print(f"  建议: 对超大用户截断或加盐 (salt) 打散")


# ============================================================
# 模块4 — 为 PyTorch 做数据准备
# ============================================================
def prepare_for_pytorch(trajectories: DataFrame) -> DataFrame:
    """
    对轨迹数据做最后的清洗和格式化, 为 PyTorch Dataset 准备。

    1. reward 归一化到 [0, 1] (原始 0.5-5.0, 归一化到 0-1)
    2. 添加 trajectory_id 唯一标识
    3. 按 userId 和时间排序输出 (方便 PyTorch 顺序读取)
    4. 分区控制: 控制 parquet 文件数量, 避免小文件过多

    面试话术:
      coalesce vs repartition:
      - coalesce(N) 只做窄依赖合并, 不触发 Shuffle, 适合减少分区
      - repartition(N) 触发全量 Shuffle, 适合均匀重分布
      这里用 coalesce 因为数据已经按 userId 有序, 无需全局打散。
    """
    print("\n" + "=" * 60)
    print("模块4 — PyTorch 数据准备")
    print("=" * 60)

    # Reward 归一化: MovieLens 评分范围 [0.5, 5.0]
    # 归一化到 [0, 1]
    trajectories = trajectories.withColumn(
        "reward_normalized",
        (F.col("reward") - 0.5) / 4.5,
    )

    # 添加唯一 ID (monotonically_increasing_id 不触发 Shuffle)
    trajectories = trajectories.withColumn(
        "trajectory_id",
        F.monotonically_increasing_id(),
    )

    # 按 userId 和 rn 排序
    trajectories = trajectories.orderBy("userId", "rn")

    # 控制输出分区数: coalesce 不触发 Shuffle（窄依赖）
    n_partitions = SPARK_CONFIG.get("spark.default.parallelism", 16)
    trajectories = trajectories.coalesce(int(n_partitions))

    n = trajectories.count()
    print(f"  最终轨迹数: {n:,}")
    print(f"  输出分区数: {n_partitions}")
    print(f"  Reward 归一化: [0.5, 5.0] → [0, 1]")

    return trajectories


# ============================================================
# 模块5 — 输出 & 执行计划展示
# ============================================================
def save_and_explain(trajectories: DataFrame):
    """
    保存轨迹 parquet, 并展示 Spark 执行计划。

    面试话术:
      explain() 是 Spark 调试利器:
      - explain()      → 物理执行计划 (已经过优化器)
      - explain(True)  → 逻辑计划 + 优化后逻辑计划 + 物理计划
      可以从中看到:
        - Project / Filter / Aggregate → 各算子
        - BroadcastHashJoin vs SortMergeJoin → Join 策略
        - Exchange → Shuffle 位置
    """
    print("\n" + "=" * 60)
    print("模块5 — 保存轨迹 & 执行计划")
    print("=" * 60)

    # 展示物理执行计划 (面试可讲)
    print("\n  === Spark 物理执行计划 (explain) ===")
    print("  (可以看到 Window → Project → Filter 的 DAG 链路)")
    trajectories.explain()

    # 保存
    trajectories.write.mode("overwrite").parquet(TRAJECTORIES_PATH)
    print(f"\n  ✅ 轨迹已保存: {TRAJECTORIES_PATH}")

    # 文件大小
    import subprocess
    try:
        result = subprocess.run(
            ["du", "-sh", TRAJECTORIES_PATH],
            capture_output=True, text=True
        )
        print(f"  文件大小: {result.stdout.strip().split()[0]}")
    except Exception:
        pass


# ============================================================
# 模块6 (Online) — 逐轮轨迹构建
# ============================================================
def build_trajectories_online(spark: SparkSession):
    """
    Online Simulation 模式: 为每一轮在线学习构建轨迹。

    读取 preprocess.py 产出的 online_windows/round{N}_train.parquet,
    为每轮构建轨迹, 保存到 output/online_windows/round{N}_trajectories.parquet。

    面试话术:
      "每轮训练数据是累积的 — 轮3的训练集 = 轮2训练集 + 窗口3数据。
       这意味着轨迹也要逐轮构建, 因为随着新数据加入,
       窗口函数构建的 state 内容会变化 (窗口内包含更近的电影)。
       这正是在线学习的真实模拟。"
    """
    online_dir = os.path.join(OUTPUT_DIR, "online_windows")

    if not os.path.exists(online_dir):
        print(f"\n❌ 找不到时间窗口数据: {online_dir}")
        print("   请先运行: python spark/preprocess.py")
        return

    # 发现所有 round 训练数据
    import glob
    train_files = sorted(glob.glob(os.path.join(online_dir, "round*_train.parquet")))
    print(f"\n[在线模式] 发现 {len(train_files)} 轮训练数据")

    for train_path in train_files:
        round_name = os.path.basename(train_path).replace("_train.parquet", "")
        traj_path = os.path.join(online_dir, f"{round_name}_trajectories.parquet")

        print(f"\n{'='*60}")
        print(f"  构建轨迹: {round_name}")
        print(f"{'='*60}")

        ratings = spark.read.parquet(train_path)
        n = ratings.count()
        n_users = ratings.select("userId").distinct().count()
        print(f"  训练评分: {n:,} 条, {n_users:,} 用户")

        if n < MIN_TRAJECTORY_LEN * n_users:
            print(f"  ⚠️ 数据不足, 跳过")
            continue

        # 构建轨迹 (复用核心逻辑)
        trajectories = build_trajectories(ratings)
        trajectories = prepare_for_pytorch(trajectories)

        # 保存
        trajectories.write.mode("overwrite").parquet(traj_path)
        print(f"  ✅ 轨迹已保存: {traj_path} ({trajectories.count():,} 条)")

    # 保存元信息
    import json
    meta = {
        "rounds": len(train_files),
        "trajectory_files": [
            os.path.basename(f).replace("_train.parquet", "_trajectories.parquet")
            for f in train_files
        ],
    }
    with open(os.path.join(online_dir, "trajectories_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ 全部轨迹构建完成 ({len(train_files)} 轮)")


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SparRL 轨迹构建")
    parser.add_argument("--online", action="store_true",
                        help="在线模拟模式: 逐轮构建轨迹")
    parser.add_argument("--round", type=int, default=None,
                        help="只构建指定轮次 (需要 --online)")
    args = parser.parse_args()

    total_t0 = time.time()
    spark = init_spark()

    try:
        if args.online:
            # Online 模式: 逐轮构建
            build_trajectories_online(spark)
        else:
            # 标准模式: 单次构建
            ratings = load_train_ratings(spark)
            trajectories = build_trajectories(ratings)
            analyze_trajectories(trajectories)
            trajectories = prepare_for_pytorch(trajectories)
            save_and_explain(trajectories)

        total_elapsed = time.time() - total_t0
        mode = "Online" if args.online else "标准"
        print(f"\n{'='*60}")
        print(f"轨迹构建完成 ({mode}模式)! 总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        print(f"{'='*60}")
        if not args.online:
            print(f"""
  面试速记:
    PARTITION BY userId ORDER BY timestamp  → 用户时序排序
    ROWS BETWEEN {STATE_SIZE} PRECEDING AND 1 PRECEDING  → 前K部做State
    collect_list(STRUCT(movieId, rating))  → 聚合为序列
    rn > {STATE_SIZE} 过滤  → 只保留完整State的样本

  产出: {TRAJECTORIES_PATH}
        → 下一步: python model/train.py (PyTorch BCQ 训练)
""")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
