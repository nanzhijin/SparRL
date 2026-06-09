"""
BCQ 训练循环
================================================================
SparRL Phase 2.4 — 南志锦 · 2026-06-09

把 Spark 产出的轨迹数据喂给 BCQ 模型训练。

训练分为两个阶段:
  Phase A: VAE 预训练 (只训练 Generator, 让它学会复现行为策略)
  Phase B: 联合训练 (VAE + Q + Perturbation 协同优化)

AutoDL 部署:
  - 自动检测 GPU
  - 支持混合精度训练 (AMP) 节省显存
  - 定期 checkpoint 防训练中断
"""

import sys
import io
import os
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import *
from model.embeddings import MovieEmbedding, MovieEmbeddingLookup, StateEncoder
from model.bcq import BCQ


# ============================================================
# 数据集 & 数据加载
# ============================================================
class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset: 从 Spark 输出的 parquet 加载轨迹数据

    每个样本:
      state_movie_ids:    (STATE_SIZE,)  int array
      state_ratings:      (STATE_SIZE,)  float array
      action_movie_id:    scalar
      reward:             scalar (normalized to [0,1])
      next_state_movie_ids: (STATE_SIZE,) int array
      next_state_ratings:   (STATE_SIZE,) float array
    """

    def __init__(self, trajectories_path: str):
        print(f"[Dataset] 从 {trajectories_path} 加载轨迹...")
        t0 = time.time()

        # 用 PyArrow 读 parquet (不需要 Spark runtime)
        self.df = pd.read_parquet(trajectories_path)

        elapsed = time.time() - t0
        print(f"[Dataset] 加载 {len(self.df):,} 条轨迹, 耗时 {elapsed:.1f}s")
        print(f"[Dataset] 列: {list(self.df.columns)}")

        # 内存预估
        mem_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"[Dataset] 内存占用: {mem_mb:.1f} MB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 转换为 tensor
        state_movie_ids = torch.tensor(
            np.array(row["state_movie_ids"]), dtype=torch.long
        )
        state_ratings = torch.tensor(
            np.array(row["state_ratings"]), dtype=torch.float32
        )
        action_movie_id = torch.tensor(row["action_movie_id"], dtype=torch.long)
        reward = torch.tensor(row["reward_normalized"], dtype=torch.float32)
        next_state_movie_ids = torch.tensor(
            np.array(row["next_state_movie_ids"]), dtype=torch.long
        )
        next_state_ratings = torch.tensor(
            np.array(row["next_state_ratings"]), dtype=torch.float32
        )

        return (
            state_movie_ids, state_ratings,
            action_movie_id, reward,
            next_state_movie_ids, next_state_ratings,
        )


def build_movie_lookup(dataset: TrajectoryDataset, df_movies_path: str):
    """
    构建 MovieEmbeddingLookup: 从电影特征 parquet 创建全量嵌入查找表

    这个查找表在整个训练过程中被共享,
    State Encoder 和 BCQ 都用它来获取电影嵌入。
    """
    print(f"\n[MovieLookup] 从 {df_movies_path} 构建电影嵌入...")

    movies_df = pd.read_parquet(df_movies_path)
    print(f"  电影数: {len(movies_df):,}")

    # 提取 genre 列
    genre_cols = sorted([c for c in movies_df.columns if c.startswith("genre_")])
    movie_genres = torch.tensor(
        movies_df[genre_cols].values, dtype=torch.float32
    )

    # 提取连续特征
    movie_years = torch.tensor(
        movies_df["release_year"].fillna(1990).values, dtype=torch.float32
    )
    movie_avg_ratings = torch.tensor(
        movies_df["avg_rating"].fillna(3.5).values, dtype=torch.float32
    )
    movie_rating_stds = torch.tensor(
        movies_df["rating_std"].fillna(0.5).values, dtype=torch.float32
    )
    movie_popularity = torch.tensor(
        movies_df["popularity_percentile"].fillna(0.5).values, dtype=torch.float32
    )

    # 构建 movieId → index 映射 (movieId 可能不连续)
    movie_ids = torch.tensor(movies_df["movieId"].values, dtype=torch.long)
    movie_id_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}

    n_movies = len(movies_df)

    lookup = MovieEmbeddingLookup(
        n_movies=n_movies,
        movie_genres=movie_genres,
        movie_years=movie_years,
        movie_avg_ratings=movie_avg_ratings,
        movie_rating_stds=movie_rating_stds,
        movie_popularity=movie_popularity,
    )

    return lookup, movie_id_to_idx


def collate_batch(batch, movie_lookup: MovieEmbeddingLookup,
                  movie_id_to_idx: dict, state_encoder: StateEncoder,
                  device: torch.device):
    """
    自定义 batch collate: 把 movie IDs 转为 embeddings,
    然后通过 StateEncoder 得到 state 向量。

    这是数据流的关键环节:
      movie IDs → movie embeddings → GRU → state vector
    """
    state_movie_ids = torch.stack([b[0] for b in batch])        # (B, K)
    state_ratings = torch.stack([b[1] for b in batch])          # (B, K)
    action_movie_id = torch.stack([b[2] for b in batch])        # (B,)
    reward = torch.stack([b[3] for b in batch]).unsqueeze(-1)   # (B, 1)
    next_state_movie_ids = torch.stack([b[4] for b in batch])   # (B, K)
    next_state_ratings = torch.stack([b[5] for b in batch])     # (B, K)

    # 映射 movieId → 连续索引
    def map_ids(ids):
        return torch.tensor(
            [[movie_id_to_idx.get(int(mid), 0) for mid in row] for row in ids],
            dtype=torch.long,
        )

    state_idx = map_ids(state_movie_ids)
    action_idx = torch.tensor(
        [movie_id_to_idx.get(int(mid), 0) for mid in action_movie_id],
        dtype=torch.long,
    )
    next_state_idx = map_ids(next_state_movie_ids)

    # 移到设备
    state_idx = state_idx.to(device)
    state_ratings = state_ratings.to(device)
    action_idx = action_idx.to(device)
    reward = reward.to(device)
    next_state_idx = next_state_idx.to(device)
    next_state_ratings = next_state_ratings.to(device)

    # 获取电影嵌入
    with torch.no_grad():
        state_embs = movie_lookup(state_idx)          # (B, K, D)
        action_emb = movie_lookup(action_idx)          # (B, D)
        next_state_embs = movie_lookup(next_state_idx) # (B, K, D)

    # State Encoder: 序列嵌入 → state 向量
    state_vec = state_encoder(state_embs, state_ratings)              # (B, state_dim)
    next_state_vec = state_encoder(next_state_embs, next_state_ratings) # (B, state_dim)

    # Done flag: 对于 MovieLens, 所有轨迹都不是 terminal
    # (用户总会继续看电影, 除非是序列最后一条 — 我们标记为 done=0)
    done = torch.zeros(reward.shape[0], 1, device=device)

    return {
        "state": state_vec,
        "action": action_emb,
        "reward": reward,
        "next_state": next_state_vec,
        "done": done,
    }


# ============================================================
# 训练器
# ============================================================
class BCQTrainer:
    """
    BCQ 训练器: 管理整个训练生命周期

    面试话术:
      "BCQ 训练分为 VAE 预训练和联合训练两个阶段。
      预训练让 VAE 先学会行为策略的分布, 否则 Q 网络
      在 VAE 还没收敛时就开始训练会导致不稳定的梯度。"
    """

    def __init__(
        self,
        model: BCQ,
        state_encoder: StateEncoder,
        movie_lookup: MovieEmbeddingLookup,
        device: torch.device,
        lr: float = LEARNING_RATE,
    ):
        self.model = model.to(device)
        self.state_encoder = state_encoder.to(device)
        self.movie_lookup = movie_lookup.to(device)
        self.device = device

        # 优化器: 分组管理
        self.optimizer_vae = optim.Adam(
            list(model.generator.parameters()) +
            list(state_encoder.parameters()) +
            list(movie_lookup.parameters()),
            lr=lr,
        )
        self.optimizer_q = optim.Adam(
            list(model.q1.parameters()) +
            list(model.q2.parameters()),
            lr=lr,
        )
        self.optimizer_perturb = optim.Adam(
            model.perturbation.parameters(), lr=lr * 0.1
        )

        # 学习率调度
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_vae, T_max=50
        )
        self.scheduler_q = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_q, T_max=50
        )

        # 混合精度
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.use_amp = (device.type == "cuda")

        # 日志
        self.metrics = defaultdict(list)

    def train_vae_pretrain(self, dataloader: DataLoader, epochs: int = 10,
                           movie_id_to_idx: dict = None):
        """
        Phase A: VAE 预训练

        只训练 VAE Generator + StateEncoder + MovieLookup,
        让 VAE 学会给定 state 时复现 action 的分布。

        这是 BCQ 的关键步骤——如果 VAE 不先收敛,
        Q 网络的梯度会破坏 VAE 的训练。
        """
        print("\n" + "=" * 60)
        print(f"Phase A: VAE 预训练 ({epochs} epochs)")
        print("=" * 60)

        if movie_id_to_idx is None:
            raise ValueError("需要 movie_id_to_idx 映射")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            self.model.train()
            self.state_encoder.train()
            self.movie_lookup.train()

            for batch_idx, raw_batch in enumerate(dataloader):
                # 手动 collate
                batch = collate_batch(
                    raw_batch, self.movie_lookup, movie_id_to_idx,
                    self.state_encoder, self.device,
                )

                self.optimizer_vae.zero_grad()

                if self.use_amp:
                    with autocast():
                        recon, mu, logvar = self.model.generator(
                            batch["state"], batch["action"]
                        )
                        vae_loss, recon_loss, kl_loss = self.model.generator.vae_loss(
                            batch["action"], recon, mu, logvar
                        )
                    self.scaler.scale(vae_loss).backward()
                    self.scaler.step(self.optimizer_vae)
                    self.scaler.update()
                else:
                    recon, mu, logvar = self.model.generator(
                        batch["state"], batch["action"]
                    )
                    vae_loss, recon_loss, kl_loss = self.model.generator.vae_loss(
                        batch["action"], recon, mu, logvar
                    )
                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.generator.parameters()) +
                        list(self.state_encoder.parameters()), GRAD_CLIP
                    )
                    self.optimizer_vae.step()

                epoch_loss += vae_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            n_batches = len(dataloader)
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches

            self.scheduler_vae.step()
            self.metrics["vae_pretrain_loss"].append(avg_loss)

            print(f"  Epoch {epoch+1:>3}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"KL: {avg_kl:.4f} | "
                  f"LR: {self.scheduler_vae.get_last_lr()[0]:.2e}")

    def train_joint(self, dataloader: DataLoader, epochs: int = 50,
                    movie_id_to_idx: dict = None, vae_weight: float = 0.5,
                    save_path: str = None):
        """
        Phase B: BCQ 联合训练

        VAE + Q-Network + Perturbation 协同优化:
          - VAE: 保持 action 约束
          - Q: 学习价值函数
          - Perturbation: 在约束内改进策略
        """
        print("\n" + "=" * 60)
        print(f"Phase B: BCQ 联合训练 ({epochs} epochs)")
        print("=" * 60)

        if movie_id_to_idx is None:
            raise ValueError("需要 movie_id_to_idx 映射")

        best_q_loss = float('inf')

        for epoch in range(epochs):
            metrics_epoch = defaultdict(float)

            self.model.train()
            self.state_encoder.train()

            for batch_idx, raw_batch in enumerate(dataloader):
                batch = collate_batch(
                    raw_batch, self.movie_lookup, movie_id_to_idx,
                    self.state_encoder, self.device,
                )

                # --- 更新 Q ---
                self.optimizer_q.zero_grad()

                if self.use_amp:
                    with autocast():
                        total_loss, loss_dict = self.model.compute_loss(
                            batch, vae_weight=vae_weight
                        )
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer_q)
                    self.scaler.update()
                else:
                    total_loss, loss_dict = self.model.compute_loss(
                        batch, vae_weight=vae_weight
                    )
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.q1.parameters()) +
                        list(self.model.q2.parameters()), GRAD_CLIP
                    )
                    self.optimizer_q.step()

                # --- 更新 VAE ---
                self.optimizer_vae.zero_grad()
                if self.use_amp:
                    with autocast():
                        recon, mu, logvar = self.model.generator(
                            batch["state"], batch["action"]
                        )
                        vae_loss, _, _ = self.model.generator.vae_loss(
                            batch["action"], recon, mu, logvar
                        )
                    self.scaler.scale(vae_loss).backward()
                    self.scaler.step(self.optimizer_vae)
                    self.scaler.update()
                else:
                    recon, mu, logvar = self.model.generator(
                        batch["state"], batch["action"]
                    )
                    vae_loss, _, _ = self.model.generator.vae_loss(
                        batch["action"], recon, mu, logvar
                    )
                    vae_loss.backward()
                    self.optimizer_vae.step()

                # --- 更新 Perturbation ---
                self.optimizer_perturb.zero_grad()
                if self.use_amp:
                    with autocast():
                        perturbed = self.model.perturbation(
                            batch["state"], batch["action"]
                        )
                        q_perturbed = self.model.q1(batch["state"], perturbed)
                        perturb_loss = -q_perturbed.mean()
                    self.scaler.scale(perturb_loss).backward()
                    self.scaler.step(self.optimizer_perturb)
                    self.scaler.update()
                else:
                    perturbed = self.model.perturbation(
                        batch["state"], batch["action"]
                    )
                    q_perturbed = self.model.q1(batch["state"], perturbed)
                    perturb_loss = -q_perturbed.mean()
                    perturb_loss.backward()
                    self.optimizer_perturb.step()

                # --- 软更新目标网络 ---
                self.model.soft_update_target()

                # 记录
                for k, v in loss_dict.items():
                    metrics_epoch[k] += v

            # Epoch 结束
            n_batches = len(dataloader)
            for k in metrics_epoch:
                metrics_epoch[k] /= n_batches

            self.scheduler_q.step()
            self.scheduler_vae.step()

            for k, v in metrics_epoch.items():
                self.metrics[k].append(v)

            q_loss = metrics_epoch["q_loss"]
            print(f"  Epoch {epoch+1:>3}/{epochs} | "
                  f"Q: {q_loss:.4f} | "
                  f"VAE: {metrics_epoch['vae_loss']:.4f} | "
                  f"Perturb: {metrics_epoch['perturb_loss']:.4f} | "
                  f"Q1_mean: {metrics_epoch['q1_mean']:.4f} | "
                  f"Target: {metrics_epoch['target_mean']:.4f}")

            # Checkpoint
            if save_path and q_loss < best_q_loss:
                best_q_loss = q_loss
                self.save(save_path)
                print(f"  ✅ Checkpoint saved (Q loss: {q_loss:.4f})")

    def save(self, path: str):
        """保存完整模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "bcq": self.model.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "movie_lookup": self.movie_lookup.state_dict(),
            "metrics": dict(self.metrics),
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["bcq"])
        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.movie_lookup.load_state_dict(checkpoint["movie_lookup"])
        self.metrics = defaultdict(list, checkpoint.get("metrics", {}))
        print(f"  Model loaded from {path}")


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SparRL BCQ 训练")
    parser.add_argument("--online", action="store_true",
                        help="在线模拟模式: 逐轮训练并保存 checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("SparRL — BCQ 训练")
    print("=" * 60)

    # 环境检测
    is_autodl, device, gpu_count = detect_autodl()
    print(f"\n[环境] AutoDL: {is_autodl}")
    print(f"[环境] Device: {device}")
    if device == "cuda":
        print(f"[环境] GPU:   {torch.cuda.get_device_name(0)}")
        print(f"[环境] VRAM:  {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"[环境] PyTorch: {torch.__version__}")

    # ─── Online Simulation 模式 ★ ───
    if args.online:
        import glob
        online_dir = os.path.join(OUTPUT_DIR, "online_windows")
        if not os.path.exists(online_dir):
            print(f"\n❌ 找不到时间窗口数据: {online_dir}")
            print("   请先运行: python spark/trajectories.py --online")
            return

        traj_files = sorted(glob.glob(os.path.join(online_dir, "round*_trajectories.parquet")))
        if not traj_files:
            print(f"\n❌ 找不到轨迹文件, 先运行: python spark/trajectories.py --online")
            return

        print(f"\n[在线模式] 逐轮训练 ({len(traj_files)} 轮)")
        print(f"[在线模式] 每轮: 累积训练数据 → 保存 checkpoint → 下一轮 warm start")

        # 初始化模型 (第一轮)
        movie_lookup, movie_id_to_idx = None, None
        state_encoder = None
        bcq_model = None

        for r, traj_path in enumerate(traj_files, start=1):
            round_name = os.path.basename(traj_path).replace("_trajectories.parquet", "")
            ckpt_path = os.path.join(online_dir, f"{round_name}_model.pt")

            print(f"\n{'='*60}")
            print(f"  在线训练 轮{r}/{len(traj_files)}: {round_name}")
            print(f"{'='*60}")

            # 加载本轮轨迹
            dataset = TrajectoryDataset(traj_path)
            dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=4 if device.type == "cuda" else 0,
                pin_memory=(device.type == "cuda"), drop_last=True,
            )
            print(f"  训练样本: {len(dataset):,}")

            # 第一轮初始化, 后续轮复用 (warm start)
            if r == 1:
                movie_lookup, movie_id_to_idx = build_movie_lookup(dataset, MOVIE_FEATURES_PATH)
                state_encoder = StateEncoder()
                bcq_model = BCQ()
                # VAE 预训练 (仅第一轮)
                trainer = BCQTrainer(bcq_model, state_encoder, movie_lookup, device)
                print(f"  模型参数: {sum(p.numel() for p in bcq_model.parameters()):,}")
                vae_epochs = 5 if is_autodl else NUM_EPOCHS_LOCAL
                trainer.train_vae_pretrain(
                    dataloader, epochs=vae_epochs, movie_id_to_idx=movie_id_to_idx
                )
            else:
                trainer = BCQTrainer(bcq_model, state_encoder, movie_lookup, device)

            # 联合训练 (每轮)
            joint_epochs = NUM_EPOCHS if is_autodl else NUM_EPOCHS_LOCAL
            trainer.train_joint(
                dataloader, epochs=joint_epochs,
                movie_id_to_idx=movie_id_to_idx,
                vae_weight=VAE_LOSS_WEIGHT,
                save_path=ckpt_path,
            )
            print(f"  ✅ round{r} 完成 → {ckpt_path}")

        print(f"\n{'='*60}")
        print(f"在线训练全部完成! ({len(traj_files)} 轮)")
        print(f"  下一步: python eval/evaluate.py --online")
        return

    # ─── 标准训练模式 ───
    # 检查数据
    if not os.path.exists(TRAJECTORIES_PATH):
        print(f"\n❌ 找不到轨迹文件: {TRAJECTORIES_PATH}")
        print("   请先运行: python spark/trajectories.py")
        return

    if not os.path.exists(MOVIE_FEATURES_PATH):
        print(f"\n❌ 找不到电影特征: {MOVIE_FEATURES_PATH}")
        print("   请先运行: python spark/preprocess.py")
        return

    # 加载轨迹数据
    dataset = TrajectoryDataset(TRAJECTORIES_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # 构建电影查找表
    movie_lookup, movie_id_to_idx = build_movie_lookup(dataset, MOVIE_FEATURES_PATH)

    # 初始化模型
    state_encoder = StateEncoder()
    bcq_model = BCQ()

    n_params = sum(p.numel() for p in bcq_model.parameters())
    n_params_enc = sum(p.numel() for p in state_encoder.parameters())
    n_params_lookup = sum(p.numel() for p in movie_lookup.parameters())
    print(f"\n[模型] BCQ 参数:    {n_params:,}")
    print(f"[模型] Encoder 参数: {n_params_enc:,}")
    print(f"[模型] Lookup 参数:  {n_params_lookup:,}")
    print(f"[模型] 总参数:       {n_params + n_params_enc + n_params_lookup:,}")

    # 训练器
    trainer = BCQTrainer(bcq_model, state_encoder, movie_lookup, device)

    # Phase A: VAE 预训练
    pretrain_epochs = 5 if is_autodl else NUM_EPOCHS_LOCAL
    trainer.train_vae_pretrain(
        dataloader, epochs=pretrain_epochs, movie_id_to_idx=movie_id_to_idx
    )

    # Phase B: 联合训练
    joint_epochs = NUM_EPOCHS if is_autodl else NUM_EPOCHS_LOCAL
    trainer.train_joint(
        dataloader, epochs=joint_epochs,
        movie_id_to_idx=movie_id_to_idx,
        vae_weight=VAE_LOSS_WEIGHT,
        save_path=MODEL_SAVE_PATH,
    )

    # 最终保存
    trainer.save(MODEL_SAVE_PATH)

    # 输出训练曲线摘要
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  Q Loss:       {trainer.metrics['q_loss'][-1]:.4f} (末)")
    print(f"  VAE Loss:     {trainer.metrics['vae_loss'][-1]:.4f} (末)")
    print(f"  Perturb Loss: {trainer.metrics['perturb_loss'][-1]:.4f} (末)")
    print(f"  Model:        {MODEL_SAVE_PATH}")
    print(f"\n  下一步: python eval/evaluate.py")


if __name__ == "__main__":
    main()
