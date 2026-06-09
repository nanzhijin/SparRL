"""
Spark 数据预处理 & 特征工程
================================================================
SparRL Phase 1.2 — 南志锦 · 2026-06-09

职责:
  1. 初始化 SparkSession（含性能配置）
  2. 加载 MovieLens 25M 原始 CSV
  3. 电影特征工程: genre one-hot / year / popularity / avg_rating
  4. 用户特征工程: avg_rating / std / count / genre_preference_vector
  5. 冷启动过滤 + 训练/测试集按用户时序划分
  6. 输出 user_features.parquet + movie_features.parquet

面试 Spark 考点:
  - Lazy Evaluation: transformations 不触发计算, action 才触发
  - 宽依赖 vs 窄依赖: groupBy → Shuffle (宽); select/filter → 无 (窄)
  - Broadcast Join: 小表 movies 广播到所有 executor
  - 列式存储: parquet 压缩 + 谓词下推
  - Spark SQL 窗口函数 (在 trajectories.py 中集中展示)
"""

import sys
import io
import os
import json
import time
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# 强制 stdout UTF-8 (Windows 兼容)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 导入全局配置
from config import *


# ============================================================
# 模块0 — Spark 环境初始化
# ============================================================
def init_spark() -> SparkSession:
    """
    初始化 SparkSession。

    面试话术:
      local[*] = 使用所有可用 CPU 核, 适合开发和单机场景.
      生产环境会写 spark://... 或 yarn.
      adaptive.enabled = Spark 3.0+ 自适应查询执行,
      会在运行时根据数据统计自动优化 Shuffle 分区数
      和 Join 策略, 这是 Spark SQL 的杀手级特性.

      SparkSession 是 2.0+ 统一入口, 包含了旧版的
      SparkContext + SQLContext + HiveContext.
    """
    builder = SparkSession.builder
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, str(value))

    spark = builder.getOrCreate()

    print(f"[Spark] 版本:       {spark.version}")
    print(f"[Spark] 应用名:     {spark.sparkContext.appName}")
    print(f"[Spark] Master:     {spark.sparkContext.master}")
    print(f"[Spark] 可用核数:   {spark.sparkContext.defaultParallelism}")
    print(f"[Spark] 执行器内存: {spark.sparkContext.getConf().get('spark.driver.memory', 'N/A')}")

    # 设置日志级别 (减少 INFO 噪音)
    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# 模块1 — 数据加载 & 基础统计
# ============================================================
def load_data(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    """
    加载 ratings.csv 和 movies.csv。

    面试话术:
      inferSchema=True 让 Spark 自动推断列类型——它会对数据
      做一次额外的采样扫描。生产环境建议显式指定 schema,
      避免推断的开销和类型错误。

      cache() 把 DataFrame 缓存到内存/磁盘, 后续多次使用时
      避免重新读取和计算。但 cache 是 lazy 的,
      需要跟一个 action 才能触发物化。
    """
    print("\n" + "=" * 60)
    print("模块1 — 数据加载")
    print("=" * 60)

    t0 = time.time()

    # 显式 schema 生产最佳实践
    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True),
    ])

    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
    ])

    ratings = spark.read.csv(
        RATINGS_PATH, header=True, schema=ratings_schema
    )
    movies = spark.read.csv(
        MOVIES_PATH, header=True, schema=movies_schema
    )

    # cache: 后续会多次用到
    ratings.cache()
    movies.cache()

    # 触发 cache 物化 + 获取统计 (一个 action 完成两件事)
    n_ratings = ratings.count()
    n_users = ratings.select("userId").distinct().count()
    n_movies = ratings.select("movieId").distinct().count()
    sparsity = n_ratings / (n_users * n_movies)

    elapsed = time.time() - t0
    print(f"  评分记录: {n_ratings:,} 行")
    print(f"  用户数:   {n_users:,}")
    print(f"  电影数:   {n_movies:,}")
    print(f"  稀疏度:   {sparsity:.6f} ({sparsity*100:.4f}%)")
    print(f"  加载耗时: {elapsed:.1f}s")

    # 评分分布 (使用 describe + 直方图)
    print("\n  评分分布:")
    ratings.select("rating").describe().show()
    ratings.groupBy("rating").count().orderBy("rating").show()

    return ratings, movies


# ============================================================
# 模块2 — 过滤冷用户 & 冷门电影
# ============================================================
def filter_cold_start(ratings: DataFrame) -> DataFrame:
    """
    过滤评分数过少的用户和电影。

    面试话术:
      groupBy + count → filter 是一个典型的窄→宽→窄转换链.
      groupBy 触发 Shuffle (宽依赖), 数据按 key 跨节点重分布.
      之后再做 broadcast join 把小表分发到各节点.

      冷启动问题: 协同过滤/RL 都依赖足够的交互数据.
      实际工业界用 content-based feature 或 exploration policy
      解决冷启动——这也是为什么我们保留了电影 genre/year
      作为 content feature.
    """
    print("\n" + "=" * 60)
    print("模块2 — 冷启动过滤")
    print("=" * 60)

    n_before = ratings.count()

    # Step 1: 过滤冷门电影
    movie_counts = ratings.groupBy("movieId").agg(
        F.count("*").alias("movie_count")
    )
    popular_movies = movie_counts.filter(
        F.col("movie_count") >= MIN_MOVIE_RATINGS
    ).select("movieId")
    ratings = ratings.join(
        F.broadcast(popular_movies), on="movieId", how="inner"
    )

    # Step 2: 过滤冷用户
    user_counts = ratings.groupBy("userId").agg(
        F.count("*").alias("user_count")
    )
    active_users = user_counts.filter(
        F.col("user_count") >= MIN_USER_RATINGS
    ).select("userId")
    ratings = ratings.join(
        F.broadcast(active_users), on="userId", how="inner"
    )

    # Step 3: 可选采样 (本地测试用)
    if SAMPLE_USERS:
        sampled_ids = ratings.select("userId").distinct().limit(SAMPLE_USERS)
        ratings = ratings.join(F.broadcast(sampled_ids), on="userId")

    n_after = ratings.count()
    n_users = ratings.select("userId").distinct().count()
    n_movies = ratings.select("movieId").distinct().count()

    print(f"  过滤前: {n_before:,} 条评分")
    print(f"  过滤后: {n_after:,} 条评分 ({n_after/n_before*100:.1f}%)")
    print(f"  活跃用户: {n_users:,}")
    print(f"  热门电影: {n_movies:,}")
    print(f"  过滤条件: 用户≥{MIN_USER_RATINGS}部, 电影≥{MIN_MOVIE_RATINGS}评分")

    return ratings


# ============================================================
# 模块3 — 电影特征工程
# ============================================================
def build_movie_features(ratings: DataFrame, movies: DataFrame) -> DataFrame:
    """
    构建每部电影的特征向量。

    特征:
      1. genre_vec:    20 维 genre multi-hot 编码 (Action, Comedy, ...)
      2. release_year: 发行年份 (从 title 正则提取)
      3. avg_rating:   该电影的平均评分
      4. rating_count: 该电影的评分次数
      5. popularity_percentile: 评分次数的分位数 (0~1)
      6. rating_std:   评分标准差 (反映争议度)

    面试话术:
      UDF vs 内置函数: Spark 内置函数 (split, regexp_extract)
      运行在 JVM 层, 比 Python UDF 快 10-100x.
      因为 Python UDF 需要序列化每一行到 Python 进程执行.

      只有在内置函数做不到时才用 Pandas UDF (Arrow 加速).
    """
    print("\n" + "=" * 60)
    print("模块3 — 电影特征工程")
    print("=" * 60)

    t0 = time.time()

    # --- 3.1 从 movies 表提取 genre multi-hot 和 year ---
    # genre pipe-separated → array → 检查每个 genre 是否出现
    genre_split = F.split(F.col("genres"), r"\|")

    for genre in MOVIELENS_GENRES:
        movies = movies.withColumn(
            f"genre_{genre.replace(' ', '_').replace('-', '_').lower()}",
            F.array_contains(genre_split, genre).cast("int"),
        )

    # 发行年份: title 格式 "Toy Story (1995)"
    movies = movies.withColumn(
        "release_year",
        F.regexp_extract(F.col("title"), r"\((\d{4})\)", 1).cast("int"),
    )
    # 无年份的用中位数填充
    median_year = movies.filter(F.col("release_year") > 0) \
        .stat.approxQuantile("release_year", [0.5], 0.01)[0]
    movies = movies.withColumn(
        "release_year",
        F.when(F.col("release_year").isNull() | (F.col("release_year") == 0),
               F.lit(int(median_year)))
         .otherwise(F.col("release_year")),
    )

    # --- 3.2 从 ratings 统计每部电影的评分指标 ---
    movie_stats = ratings.groupBy("movieId").agg(
        F.mean("rating").alias("avg_rating"),
        F.stddev("rating").alias("rating_std"),
        F.count("rating").alias("rating_count"),
    )

    # 填充分组后可能为 null 的 stddev (只有1条评分时)
    movie_stats = movie_stats.fillna({"rating_std": 0.0})

    # 分位数: 用窗口函数计算 percentile rank
    window_spec = Window.orderBy("rating_count")
    movie_stats = movie_stats.withColumn(
        "popularity_percentile",
        F.percent_rank().over(window_spec),
    )

    # --- 3.3 合并 genre 特征 + 统计特征 ---
    # broadcast join: movies 表足够小可以广播
    movie_features = movies.join(
        F.broadcast(movie_stats), on="movieId", how="inner"
    )

    elapsed = time.time() - t0
    print(f"  电影特征维度: {len(movie_features.columns)} 列")
    print(f"  构建耗时: {elapsed:.1f}s")
    print(f"\n  特征片段 (前5部电影):")
    movie_features.select(
        "movieId", "title", "release_year", "avg_rating",
        "rating_count", "popularity_percentile",
        *[c for c in movie_features.columns if c.startswith("genre_")]
    ).show(5, truncate=False)

    return movie_features


# ============================================================
# 模块4 — 用户特征工程
# ============================================================
def build_user_features(ratings: DataFrame, movie_features: DataFrame) -> DataFrame:
    """
    构建每个用户的全局统计特征。

    特征:
      1. avg_rating:      该用户的平均评分 (反映手松/手紧)
      2. rating_std:      评分标准差 (反映区分度)
      3. rating_count:    评分总数
      4. active_days:     评分时间跨度
      5. genre_pref:      20维 genre 偏好向量
         (avg_rating per genre, 反映用户的 genre 口味)

    面试话术:
      这里的 genre_pref 是一个例子, 展示 Spark 怎么处理
      "用户对某类电影的偏好" 这种需要 join + 聚合的计算.

      计算思路:
      ratings join movies (带 genre 标记)
      → groupBy userId, 对每个 genre 列取 avg(rating)
      → 得到 user-genre 偏好矩阵
    """
    print("\n" + "=" * 60)
    print("模块4 — 用户特征工程")
    print("=" * 60)

    t0 = time.time()

    # --- 4.1 用户基础统计 ---
    user_basic = ratings.groupBy("userId").agg(
        F.mean("rating").alias("avg_rating"),
        F.stddev("rating").alias("rating_std"),
        F.count("rating").alias("rating_count"),
        F.min("timestamp").alias("first_ts"),
        F.max("timestamp").alias("last_ts"),
    ).fillna({"rating_std": 0.0})

    # 活跃天数
    user_basic = user_basic.withColumn(
        "active_days",
        (F.col("last_ts") - F.col("first_ts")) / 86400.0,
    )

    # --- 4.2 用户 genre 偏好 ---
    # 关联电影 genre 标签
    genre_cols = [c for c in movie_features.columns if c.startswith("genre_")]
    ratings_with_genre = ratings.join(
        F.broadcast(movie_features.select("movieId", *genre_cols)),
        on="movieId",
    )

    # 用户对每个 genre 的评分偏好
    # 注意: 这里用的是 genre 0/1 → 有该标签的电影的评分均值
    genre_agg_exprs = []
    for gc in genre_cols:
        short_name = gc.replace("genre_", "pref_")
        genre_agg_exprs.append(
            F.avg(F.when(F.col(gc) == 1, F.col("rating"))).alias(short_name)
        )

    user_genre_pref = ratings_with_genre.groupBy("userId").agg(
        *genre_agg_exprs
    )
    # 填充 null→0 (用户可能没评过某类电影)
    for gc in genre_cols:
        short_name = gc.replace("genre_", "pref_")
        user_genre_pref = user_genre_pref.fillna({short_name: 0.0})

    # --- 4.3 合并 ---
    user_features = user_basic.join(user_genre_pref, on="userId", how="inner")

    elapsed = time.time() - t0
    print(f"  用户特征维度: {len(user_features.columns)} 列")
    print(f"  构建耗时: {elapsed:.1f}s")
    print(f"\n  特征片段 (前5个用户):")
    user_features.select(
        "userId", "avg_rating", "rating_std", "rating_count", "active_days"
    ).show(5, truncate=False)

    # 分布洞察
    print("\n  用户评分次数分布:")
    user_features.select("rating_count").describe().show()
    print("\n  用户平均分分布:")
    user_features.select("avg_rating").describe().show()

    return user_features


# ============================================================
# 模块5a — 训练/测试集划分 (用户内时序)
# ============================================================
def split_train_test(ratings: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    按用户时序划分训练/测试集 (user-level temporal split)。

    面试话术:
      推荐系统的 train/test split 不能随机打散 — 必须按时序!
      因为推荐是在"今天"预测"明天",
      用未来数据训练会泄露信息 (data leakage).

      这里用 window function:
        ROW_NUMBER() OVER (PARTITION BY userId ORDER BY timestamp)
      给每个用户的评分按时间排序, 前 80% 作训练, 后 20% 作测试.

      这是 Spark 窗口函数的经典用法 — PARTITION BY 只做分区内排序,
      不触发 Shuffle (数据已经按 userId 分好区的前提下).
    """
    print("\n" + "=" * 60)
    print("模块5a — 训练/测试集划分 (按用户时序)")
    print("=" * 60)

    # 窗口函数: 用户内按时间排序
    window_spec = Window.partitionBy("userId").orderBy("timestamp")

    ratings_with_rn = ratings.withColumn(
        "row_num", F.row_number().over(window_spec)
    ).withColumn(
        "total_count", F.count("*").over(Window.partitionBy("userId"))
    )

    # 前 TRAIN_TEST_SPLIT 比例作训练
    train = ratings_with_rn.filter(
        F.col("row_num") <= F.col("total_count") * TRAIN_TEST_SPLIT
    ).drop("row_num", "total_count")

    test = ratings_with_rn.filter(
        F.col("row_num") > F.col("total_count") * TRAIN_TEST_SPLIT
    ).drop("row_num", "total_count")

    train_n = train.count()
    test_n = test.count()
    train_users = train.select("userId").distinct().count()
    test_users = test.select("userId").distinct().count()

    print(f"  训练集: {train_n:,} 条评分, {train_users:,} 用户")
    print(f"  测试集: {test_n:,} 条评分, {test_users:,} 用户")
    print(f"  训练占比: {train_n/(train_n+test_n)*100:.1f}%")

    return train, test


# ============================================================
# 模块5b — 全局时间窗口切割 (Online Simulation) ★
# ============================================================
def split_by_global_time_window(
    ratings: DataFrame,
    n_windows: int = N_TIME_WINDOWS,
) -> list[dict]:
    """
    按全局时间戳把完整评级数据切成 N 个等长窗口，
    模拟在线部署的持续学习过程。

    窗口逻辑:
      全量时间轴 [t_min ────────────────── t_max]
                       │    │    │    │
                       W0   W1   W2   W3

      在线模拟:
        轮1: 训练=W0       测试=W1
        轮2: 训练=W0+W1    测试=W2
        轮3: 训练=W0+W1+W2 测试=W3
        ...

    面试话术 — 为什么这算"在线学习":
      "在线学习的核心是'用过去预测未来, 不断加入新数据'。
       我用全局时间窗口切割模拟了这一点:
       - 每轮训练只能看到'到目前为止'的全部历史
       - 测试永远是训练窗口之后的'未来'数据
       - 这跟真实部署的 incremental learning 流程完全一致。

       它跟普通 train/test split 的区别:
       普通 split 只评估一个时间点,
       时间窗口评估画出的是学习曲线——
       NDCG 随'看到更多历史'的变化趋势,
       可以判断模型还需要多少数据、有没有饱和。"

    Spark 实现要点:
      - 全局 timestamp 的 min/max 是一次 action (触发扫描)
      - 每个窗口的 filter 是窄依赖 (无 Shuffle)
      - 窗口内训练集可以 cumsum 合并 (union → 不触发 Shuffle)
    """
    print("\n" + "=" * 60)
    print("模块5b — 全局时间窗口切割 (Online Simulation)")
    print("=" * 60)

    if not ONLINE_SIMULATION:
        print("  ⏭️  已禁用 (ONLINE_SIMULATION=False)")
        return []

    # Step 1: 获取全局时间范围
    ts_stats = ratings.agg(
        F.min("timestamp").alias("t_min"),
        F.max("timestamp").alias("t_max"),
        F.count("*").alias("total_count"),
    ).collect()[0]

    t_min, t_max, total_count = ts_stats["t_min"], ts_stats["t_max"], ts_stats["total_count"]
    window_size = (t_max - t_min) / n_windows

    print(f"  全量时间范围: {t_min} → {t_max}")
    print(f"  窗口数: {n_windows}")
    print(f"  每窗口时长: {window_size / 86400:.0f} 天 ({window_size:.0f}s)")

    # MovieLens 时间戳是 Unix 秒
    t_min_dt = pd.Timestamp.utcfromtimestamp(t_min)
    t_max_dt = pd.Timestamp.utcfromtimestamp(t_max)
    print(f"  日期范围: {t_min_dt.date()} → {t_max_dt.date()}")

    # Step 2: 预缓存 (避免每次 filter 都重新扫描)
    ratings_cached = ratings.cache()
    ratings_cached.count()  # 触发 cache

    # Step 3: 构建窗口
    windows = []
    window_boundaries = []

    for w in range(n_windows):
        w_start = t_min + window_size * w
        w_end = t_min + window_size * (w + 1)
        window_boundaries.append((w_start, w_end))

        # 窗口内的评分
        w_data = ratings_cached.filter(
            (F.col("timestamp") >= w_start) & (F.col("timestamp") < w_end)
        )
        w_count = w_data.count()
        w_users = w_data.select("userId").distinct().count()

        window_info = {
            "window_id": w,
            "t_start": w_start,
            "t_end": w_end,
            "date_start": pd.Timestamp.utcfromtimestamp(w_start).date(),
            "date_end": pd.Timestamp.utcfromtimestamp(w_end).date(),
            "n_ratings": w_count,
            "n_users": w_users,
        }
        windows.append(window_info)

        pct = w_count / total_count * 100
        print(f"  W{w}: {window_info['date_start']} → {window_info['date_end']}  "
              f"{w_count:>10,} 条评分 ({pct:5.1f}%)  {w_users:>6,} 用户")

    # Step 4: 构建在线学习轮次 (cumulative training + next window testing)
    print(f"\n  在线学习轮次:")
    online_rounds = []

    for r in range(1, n_windows):
        # 训练集: 窗口 0 .. r-1 (累积)
        train_ids = list(range(r))
        train_data = ratings_cached.filter(
            (F.col("timestamp") >= window_boundaries[0][0]) &
            (F.col("timestamp") < window_boundaries[r-1][1])
        )
        train_n = train_data.count()

        # 测试集: 窗口 r (未来)
        test_data = ratings_cached.filter(
            (F.col("timestamp") >= window_boundaries[r][0]) &
            (F.col("timestamp") < window_boundaries[r][1])
        )
        test_n = test_data.count()

        round_info = {
            "round": r,
            "train_windows": train_ids,
            "test_window": r,
            "train_n": train_n,
            "test_n": test_n,
            "train_start_date": windows[0]["date_start"],
            "train_end_date": windows[r-1]["date_end"],
            "test_date": windows[r]["date_start"],
        }
        online_rounds.append(round_info)

        print(f"    轮{r}: 训练=W0..W{r-1} ({train_n:>10,}条) → 测试=W{r} ({test_n:>10,}条)  "
              f"日期截止={windows[r-1]['date_end']}, 测试={windows[r]['date_start']}")

    # Step 5: 保存窗口元数据 (不用 Spark, 直接写 JSON)
    import json
    metadata = {
        "t_min": int(t_min), "t_max": int(t_max), "n_windows": n_windows,
        "t_min_date": str(t_min_dt.date()), "t_max_date": str(t_max_dt.date()),
        "windows": [
            {"id": w["window_id"], "date_start": str(w["date_start"]),
             "date_end": str(w["date_end"]), "n_ratings": int(w["n_ratings"])}
            for w in windows
        ],
    }
    meta_path = os.path.join(OUTPUT_DIR, "time_windows_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n  窗口元数据已保存: {meta_path}")

    ratings_cached.unpersist()
    return {
        "windows": windows,
        "online_rounds": online_rounds,
        "t_min": t_min,
        "t_max": t_max,
        "window_size": window_size,
    }


# ============================================================
# 模块6 — 输出 & 总览
# ============================================================
def save_and_summary(
    ratings_train: DataFrame,
    ratings_test: DataFrame,
    movie_features: DataFrame,
    user_features: DataFrame,
    online_info: dict = None,
    ratings_full: DataFrame = None,
):
    """保存 parquet 并打印总览（含时间窗口数据）"""
    print("\n" + "=" * 60)
    print("模块6 — 输出 parquet 文件")
    print("=" * 60)

    # 列式存储 parquet: 压缩率高, 带 schema, 支持谓词下推
    movie_features.write.mode("overwrite").parquet(MOVIE_FEATURES_PATH)
    user_features.write.mode("overwrite").parquet(USER_FEATURES_PATH)
    ratings_train.write.mode("overwrite").parquet(
        os.path.join(OUTPUT_DIR, "ratings_train.parquet")
    )
    ratings_test.write.mode("overwrite").parquet(
        os.path.join(OUTPUT_DIR, "ratings_test.parquet")
    )

    print(f"  ✅ {MOVIE_FEATURES_PATH}")
    print(f"  ✅ {USER_FEATURES_PATH}")
    print(f"  ✅ ratings_train.parquet")
    print(f"  ✅ ratings_test.parquet")

    # ★ 时间窗口数据: 保存每轮训练/测试的 parquet
    if online_info and ratings_full is not None:
        print(f"\n  --- 时间窗口 (Online Simulation) ---")
        online_dir = os.path.join(OUTPUT_DIR, "online_windows")
        os.makedirs(online_dir, exist_ok=True)

        for rnd in online_info["online_rounds"]:
            r = rnd["round"]
            t_start = online_info["windows"][0]["t_start"]
            t_train_end = online_info["windows"][r-1]["t_end"]
            t_test_start = online_info["windows"][r]["t_start"]
            t_test_end = online_info["windows"][r]["t_end"]

            # 训练: 窗口0→r-1 累积
            train_round = ratings_full.filter(
                (F.col("timestamp") >= t_start) & (F.col("timestamp") < t_train_end)
            )
            train_path = os.path.join(online_dir, f"round{r}_train.parquet")
            train_round.write.mode("overwrite").parquet(train_path)
            print(f"  ✅ round{r}_train.parquet — {rnd['train_n']:,} 条")

            # 测试: 窗口r
            test_round = ratings_full.filter(
                (F.col("timestamp") >= t_test_start) & (F.col("timestamp") < t_test_end)
            )
            test_path = os.path.join(online_dir, f"round{r}_test.parquet")
            test_round.write.mode("overwrite").parquet(test_path)
            print(f"  ✅ round{r}_test.parquet — {rnd['test_n']:,} 条")

    print("\n" + "=" * 60)
    print("数据管线完成 — 总览")
    print("=" * 60)

    online_section = ""
    if online_info and online_info.get("online_rounds"):
        n = len(online_info["online_rounds"])
        online_section = f"""
  时间窗口 Online Simulation:
    {n} 轮在线学习, 数据在 output/online_windows/
    每轮: round{r}_train.parquet + round{r}_test.parquet
    用法: python model/train.py --online (逐轮训练+评估 → 学习曲线)
"""

    print(f"""
  产出文件:
    1. movie_features.parquet  — {len(movie_features.columns)} 维特征 (genre + year + stats)
    2. user_features.parquet   — {len(user_features.columns)} 维特征 (偏好 + 行为统计)
    3. ratings_train.parquet   — 训练集评分 (user-level split)
    4. ratings_test.parquet    — 测试集评分 (user-level split){online_section}
  下一步:
    spark/trajectories.py — 用窗口函数构建 RL 轨迹
    (时间窗口模式下: python spark/trajectories.py --online)

  Spark 考点回顾:
    ✅ Lazy Evaluation — 所有转换在 action 时才计算
    ✅ 宽依赖 — groupBy 触发 Shuffle
    ✅ Broadcast Join — 小表 movies 广播到各节点
    ✅ 窗口函数 — ROW_NUMBER / PERCENT_RANK / 时间分区
    ✅ Parquet — 列存、压缩、schema 自包含
    ✅ 全局时间切割 — 全量扫描 min/max + cache 复用
""")


# ============================================================
# 主入口
# ============================================================
def main():
    total_t0 = time.time()

    spark = init_spark()

    try:
        # 模块1: 加载数据
        ratings, movies = load_data(spark)

        # 模块2: 冷启动过滤
        ratings = filter_cold_start(ratings)

        # 模块3: 电影特征
        movie_features = build_movie_features(ratings, movies)

        # 模块4: 用户特征
        user_features = build_user_features(ratings, movie_features)

        # ★ 模块5b: 全局时间窗口切割 (在 user-level split 之前做,
        #   因为 online simulation 需要全量 ratings 来计算时间边界)
        online_info = split_by_global_time_window(ratings)

        # 模块5a: 用户级训练/测试划分
        ratings_train, ratings_test = split_train_test(ratings)

        # 模块6: 输出 (含时间窗口 parquet)
        save_and_summary(
            ratings_train, ratings_test, movie_features, user_features,
            online_info=online_info, ratings_full=ratings,
        )

        total_elapsed = time.time() - total_t0
        print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
