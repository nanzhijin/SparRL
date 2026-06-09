"""
SparRL 全局配置
============================================================
Spark + Offline RL (BCQ) 推荐系统
南志锦 · 2026-06-09

所有路径、超参数、开关集中管理。
AutoDL 部署时只需修改 DATA_DIR 和 OUTPUT_DIR。
"""

import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-25m", "ml-25m")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
GENOME_SCORES_PATH = os.path.join(DATA_DIR, "genome-scores.csv")
GENOME_TAGS_PATH = os.path.join(DATA_DIR, "genome-tags.csv")

# Spark 中间输出
USER_FEATURES_PATH = os.path.join(OUTPUT_DIR, "user_features.parquet")
MOVIE_FEATURES_PATH = os.path.join(OUTPUT_DIR, "movie_features.parquet")
TRAJECTORIES_PATH = os.path.join(OUTPUT_DIR, "trajectories.parquet")

# PyTorch 模型保存
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "bcq_model.pt")
EMBEDDING_SAVE_PATH = os.path.join(OUTPUT_DIR, "movie_embeddings.pt")

# ============================================================
# Spark 配置
# ============================================================
SPARK_CONFIG = {
    "spark.master": "local[*]",          # AutoDL: 改为 "spark://..."
    "spark.app.name": "SparRL",
    "spark.driver.memory": "8g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.default.parallelism": "16",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
}

# ============================================================
# 数据过滤
# ============================================================
MIN_USER_RATINGS = 10       # 最少评分次数，过滤冷用户
MIN_MOVIE_RATINGS = 5       # 最少被评次数，过滤冷门电影
SAMPLE_USERS = None         # None=全量，设整数=采样N用户（本地测试用）
TRAIN_TEST_SPLIT = 0.8      # 按用户时序划分：前80%作训练轨迹，后20%作测试

# Online Simulation: 全局时间窗口切割
# 把全量数据按全局时间戳切成 N 个等长窗口，
# 模拟"持续在线学习"：窗口1训练→窗口2验证→窗口1+2训练→窗口3验证...
ONLINE_SIMULATION = True        # 是否启用时间窗口在线模拟
N_TIME_WINDOWS = 4             # 窗口数量
INITIAL_WINDOW_RATIO = 0.25    # 前 25% 时间作为初始训练集的"预热窗口"

# ============================================================
# 轨迹构建参数
# ============================================================
STATE_SIZE = 10             # state = 用户最近的 K 部电影
ACTION_DIM = 64             # 电影嵌入向量维度
MIN_TRAJECTORY_LEN = STATE_SIZE + 1  # 至少要有 K+1 条评分才能构成一个训练样本

# ============================================================
# 电影特征工程
# ============================================================
# MovieLens 的 20 个 genre（按字母序）
MOVIELENS_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)",
]
N_GENRES = len(MOVIELENS_GENRES)

# 使用 genome tags 的数量（Top-N 最相关标签）
N_GENOME_TAGS = 50
GENOME_EMB_DIM = 16          # 每个标签嵌入维度

# ============================================================
# RL 训练超参数（BCQ）
# ============================================================
# 模型结构
HIDDEN_DIM = 256             # 隐层维度
STATE_DIM = 128              # GRU 输出的状态向量维度
NUM_GRU_LAYERS = 2           # GRU 层数

# VAE (Generator)
LATENT_DIM = 32              # VAE 隐变量 z 的维度

# BCQ 特定
N_ACTIONS_SAMPLE = 100       # 从 VAE 采样候选动作的数量
PERTURBATION_PHI = 0.05      # 扰动幅度上限（控制"改进"的保守度）

# 训练
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
GAMMA = 0.99                 # 折扣因子（推荐场景可以设高一些）
TAU = 0.005                  # 目标网络软更新系数
NUM_EPOCHS = 50
NUM_EPOCHS_LOCAL = 5         # 本地快速验证用的 epoch 数
GRAD_CLIP = 1.0

# VAE 损失权重
VAE_LOSS_WEIGHT = 0.5

# Double Q-Learning
USE_DOUBLE_Q = True

# ============================================================
# 评估配置
# ============================================================
EVAL_K_VALUES = [5, 10, 20]           # Top-K 评估
EVAL_USER_SAMPLE = 1000               # 评估时采样用户数
RANDOM_SEED = 42

# ============================================================
# AutoDL 配置（部署时自动检测）
# ============================================================
def detect_autodl():
    """检测是否在 AutoDL 环境"""
    is_autodl = os.path.exists("/root/autodl-tmp") or "AUTODL" in os.environ
    if is_autodl:
        # GPU 检测
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        GPU_COUNT = torch.cuda.device_count() if DEVICE == "cuda" else 0
    else:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        GPU_COUNT = torch.cuda.device_count() if DEVICE == "cuda" else 0
    return is_autodl, DEVICE, GPU_COUNT

if __name__ == "__main__":
    is_autodl, device, gpu_count = detect_autodl()
    print(f"AutoDL: {is_autodl}")
    print(f"Device: {device}")
    print(f"GPU count: {gpu_count}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"MovieLens genres: {N_GENRES}")
