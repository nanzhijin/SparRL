"""
SparRL 端到端管线
================================================================
一键运行 Spark 数据管线 + PyTorch BCQ 训练 + 评估

用法:
  # 本地快速验证 (小规模采样)
  python run_pipeline.py --mode local --sample 5000

  # AutoDL 全量训练
  python run_pipeline.py --mode full

  # 仅 Spark 部分
  python run_pipeline.py --mode spark-only

  # 仅训练 (已有轨迹数据)
  python run_pipeline.py --mode train-only
"""

import sys
import io
import os
import time
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def run_spark_pipeline(sample_users=None):
    """运行完整 Spark 数据管线"""
    print("\n" + "█" * 60)
    print("█  Phase 1: Spark 数据管线")
    print("█" * 60)

    # 修改采样设置
    import config
    if sample_users:
        config.SAMPLE_USERS = sample_users
        print(f"\n⚙️  采样模式: {sample_users} 用户")

    t0 = time.time()

    # 1. 特征工程
    print("\n--- Step 1/2: 特征工程 ---")
    import spark.preprocess
    spark.preprocess.main()

    # 2. 轨迹构建
    print("\n--- Step 2/2: 轨迹构建 ---")
    import spark.trajectories
    spark.trajectories.main()

    elapsed = time.time() - t0
    print(f"\n✅ Spark 管线完成! 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return elapsed


def run_training():
    """运行 BCQ 训练"""
    print("\n" + "█" * 60)
    print("█  Phase 2: BCQ 训练")
    print("█" * 60)

    t0 = time.time()

    import model.train
    model.train.main()

    elapsed = time.time() - t0
    print(f"\n✅ BCQ 训练完成! 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return elapsed


def run_evaluation():
    """运行模型评估"""
    print("\n" + "█" * 60)
    print("█  Phase 3: 模型评估")
    print("█" * 60)

    t0 = time.time()

    import eval.evaluate
    eval.evaluate.main()

    elapsed = time.time() - t0
    print(f"\n✅ 评估完成! 耗时: {elapsed:.1f}s")
    return elapsed


def run_ope():
    """运行 OPE 离线策略评估"""
    print("\n" + "█" * 60)
    print("█  Phase 3b: OPE 离线策略评估")
    print("█" * 60)

    t0 = time.time()

    import spark.ope
    spark.ope.main()

    elapsed = time.time() - t0
    print(f"\n✅ OPE 完成! 耗时: {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="SparRL 端到端管线")
    parser.add_argument(
        "--mode", type=str, default="local",
        choices=["local", "full", "spark-only", "train-only", "eval-only"],
        help="运行模式 (在线模拟请直接使用各脚本的 --online 参数)"
    )
    parser.add_argument(
        "--sample", type=int, default=5000,
        help="本地模式采样用户数"
    )
    parser.add_argument(
        "--skip-ope", action="store_true",
        help="跳过 OPE 评估"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SparRL — Spark + Offline RL 推荐系统")
    print("南志锦 · 2026-06-09")
    print("=" * 60)
    print(f"\n  模式:  {args.mode}")
    if args.mode == "local":
        print(f"  采样:  {args.sample} 用户")

    total_t0 = time.time()

    if args.mode in ("local", "full", "spark-only"):
        sample = args.sample if args.mode == "local" else None
        run_spark_pipeline(sample_users=sample)

    if args.mode in ("local", "full", "train-only"):
        # 检查轨迹文件
        from config import TRAJECTORIES_PATH
        if not os.path.exists(TRAJECTORIES_PATH):
            print(f"\n❌ 找不到轨迹文件: {TRAJECTORIES_PATH}")
            print("   请先运行 Spark 管线: python run_pipeline.py --mode spark-only")
            return
        run_training()

    if args.mode in ("local", "full", "eval-only"):
        from config import MODEL_SAVE_PATH
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"\n❌ 找不到模型: {MODEL_SAVE_PATH}")
            print("   请先运行训练: python run_pipeline.py --mode train-only")
            return
        run_evaluation()

    if args.mode in ("full",) and not args.skip_ope:
        run_ope()

    total_elapsed = time.time() - total_t0
    print("\n" + "█" * 60)
    print(f"█  SparRL 全管线完成! 总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print("█" * 60)

    # 下一句话
    print("""
  面试准备好了吗? → 打开 README.md 背诵面试叙事
  模型调优?       → 修改 config.py 的超参数
  消融实验?       → 注释/开启 BCQ 各组件
""")


if __name__ == "__main__":
    main()
