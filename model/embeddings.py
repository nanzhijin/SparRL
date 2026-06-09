"""
电影 & 用户嵌入层
================================================================
SparRL Phase 2.1 — 南志锦 · 2026-06-09

把 Spark 产出的离散特征变成 PyTorch 可训练的稠密向量。

设计思路:
  电影嵌入 = 内容特征 (genre + year) + 协同特征 (avg_rating + popularity)
  这样即使冷门电影也能靠 content feature 有合理的嵌入,
  而不是纯 collaborative 那样对新电影完全无知。

  在整个 BCQ 训练流水线里, 电影嵌入是共享的——
  State Encoder 用, Generator 也用, Q-Network 也用。
"""

import torch
import torch.nn as nn
import numpy as np

from config import (
    N_GENRES, ACTION_DIM, HIDDEN_DIM, GENOME_EMB_DIM, MOVIELENS_GENRES,
)


class MovieEmbedding(nn.Module):
    """
    电影特征 → 稠密嵌入向量

    输入:
      genres:         (batch, N_GENRES) multi-hot
      release_year:   (batch,) 标量, 1900-2020
      avg_rating:     (batch,) 标量
      rating_std:     (batch,) 标量
      popularity_pct: (batch,) 标量

    输出:
      movie_emb:      (batch, ACTION_DIM)

    设计原则:
      - genre 和 year 用 embedding 捕获非线性交互
      - 统计特征直接拼入, 保持线性通路
      - 最终投影到统一维度, 方便后续 State Encoder / BCQ 复用
    """

    def __init__(self, n_genres=N_GENRES, action_dim=ACTION_DIM):
        super().__init__()

        # Genre: multi-hot → dense
        self.genre_proj = nn.Sequential(
            nn.Linear(n_genres, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Year: 分桶嵌入 (每 10 年一桶, ~12 个桶)
        self.year_min = 1900
        self.year_max = 2030
        self.year_bucket_size = 10
        n_year_buckets = (self.year_max - self.year_min) // self.year_bucket_size + 1
        self.year_embed = nn.Embedding(n_year_buckets, 8)

        # 连续统计特征 → 小嵌入
        self.stats_proj = nn.Sequential(
            nn.Linear(3, 16),  # avg_rating, rating_std, popularity_pct
            nn.ReLU(),
        )

        # 合并投影
        self.output_proj = nn.Sequential(
            nn.Linear(32 + 8 + 16, action_dim),
            nn.LayerNorm(action_dim),
        )

    def _bucket_year(self, year: torch.Tensor) -> torch.Tensor:
        """年份 → 桶索引, 超出范围的截断"""
        year = year.clamp(self.year_min, self.year_max)
        bucket = ((year - self.year_min) / self.year_bucket_size).long()
        return bucket

    def forward(
        self,
        genres: torch.Tensor,
        release_year: torch.Tensor,
        avg_rating: torch.Tensor,
        rating_std: torch.Tensor,
        popularity_pct: torch.Tensor,
    ) -> torch.Tensor:
        # Genre embedding
        genre_emb = self.genre_proj(genres)  # (B, 32)

        # Year embedding
        year_bucket = self._bucket_year(release_year)  # (B,)
        year_emb = self.year_embed(year_bucket)  # (B, 8)

        # Stats embedding
        stats = torch.stack([avg_rating, rating_std, popularity_pct], dim=-1)  # (B, 3)
        stats_emb = self.stats_proj(stats)  # (B, 16)

        # Concatenate & project
        combined = torch.cat([genre_emb, year_emb, stats_emb], dim=-1)  # (B, 56)
        movie_emb = self.output_proj(combined)  # (B, ACTION_DIM)

        return movie_emb


class MovieEmbeddingLookup(nn.Module):
    """
    全量电影嵌入查找表

    为了避免每个 batch 都重新计算所有电影的嵌入,
    在训练前一次性计算所有电影嵌入并存在 lookup table 里。

    训练时: 根据 movieId 直接 index → O(1)
    """

    def __init__(
        self,
        n_movies: int,
        movie_genres: torch.Tensor,       # (n_movies, N_GENRES)
        movie_years: torch.Tensor,        # (n_movies,)
        movie_avg_ratings: torch.Tensor,  # (n_movies,)
        movie_rating_stds: torch.Tensor,  # (n_movies,)
        movie_popularity: torch.Tensor,   # (n_movies,)
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.n_movies = n_movies
        self.action_dim = action_dim

        # 计算全量电影嵌入
        self.embedder = MovieEmbedding(action_dim=action_dim)

        with torch.no_grad():
            all_embs = self.embedder(
                movie_genres,
                movie_years,
                movie_avg_ratings,
                movie_rating_stds,
                movie_popularity,
            )  # (n_movies, ACTION_DIM)

        # 注册为 buffer (不参与梯度, 但随模型保存)
        self.register_buffer("embeddings", all_embs)
        # 添加一个可训练的偏移量 (微调嵌入)
        self.emb_offset = nn.Parameter(torch.zeros(n_movies, action_dim))
        nn.init.uniform_(self.emb_offset, -0.01, 0.01)

    def forward(self, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            movie_ids: (batch,) or (batch, seq_len) LongTensor
        Returns:
            embeddings: (*, ACTION_DIM)
        """
        base_emb = self.embeddings[movie_ids]
        offset = self.emb_offset[movie_ids]
        return base_emb + offset

    def get_all_embeddings(self) -> torch.Tensor:
        """获取所有电影嵌入 (用于最近邻搜索)"""
        return self.embeddings + self.emb_offset


class StateEncoder(nn.Module):
    """
    用户行为序列 → 状态向量

    输入: 用户最近 K 部电影的嵌入 + 对应评分
    输出: 状态向量 (表征用户当前的口味和上下文)

    架构: 双向 GRU, 最后的隐藏状态作为 state 表示。

    面试话术:
      "GRU 比 LSTM 参数少 1/4, 适合短序列推荐场景。
       双向可以捕获序列前后的 context,
       最后拼接 forward+backward 的最后一层 hidden state。"

      "为什么不直接 avg pooling?
       因为评分序列有时序依赖——先看了什么后看了什么
       反映了用户的兴趣漂移。GRU 可以学到:
       刚看完恐怖片后推什么? vs 刚看完喜剧后推什么?"
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        state_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 评分编码: scalar → small vector
        self.rating_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )

        # 输入 = 电影嵌入 + 评分嵌入
        input_dim = action_dim + 16

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # 双向 → 输出 = hidden_dim
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )

        # 最后的投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        movie_embs: torch.Tensor,  # (batch, seq_len, action_dim)
        ratings: torch.Tensor,     # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Args:
            movie_embs: 序列中的电影嵌入
            ratings:    对应的评分 (normalized to [0,1])
        Returns:
            state:       (batch, state_dim)
        """
        batch_size, seq_len, _ = movie_embs.shape

        # 评分编码
        rating_emb = self.rating_embed(ratings.unsqueeze(-1))  # (B, L, 16)

        # 拼接
        gru_input = torch.cat([movie_embs, rating_emb], dim=-1)  # (B, L, input_dim)

        # GRU
        gru_out, h_n = self.gru(gru_input)  # gru_out: (B, L, H), h_n: (4, B, H/2)

        # 拼接最后一层双向 hidden → (B, HIDDEN_DIM)
        # h_n shape: (num_layers * 2, batch, hidden_dim//2)
        last_forward = h_n[-2, :, :]  # forward 最后一层
        last_backward = h_n[-1, :, :]  # backward 最后一层
        final_hidden = torch.cat([last_forward, last_backward], dim=-1)  # (B, HIDDEN_DIM)

        # 投影
        state = self.output_proj(final_hidden)  # (B, state_dim)

        return state
