"""
BCQ (Batch-Constrained Deep Q-Learning) 模型
================================================================
SparRL Phase 2.3 — 南志锦 · 2026-06-09

BCQ 是 offline RL 的里程碑算法, 由 Fujimoto et al. (ICML 2019) 提出。

核心问题:
  标准 DQN 在 offline 数据上直接训练会崩溃——
  因为 Q 网络会对"数据中没出现过"的 action 给出荒谬的高估值
  (extrapolation error / distribution shift)。

BCQ 的解法:
  1. VAE Generator G(s) — 学习行为策略的 action 分布,
     只生成"历史上出现过的"候选动作
  2. Perturbation ξ(s,a) — 在候选动作上做小幅改进 (≤ Φ)
  3. Q-Network — 在候选动作中选出最好的

  最终策略: π(s) = argmax_{a = a' + ξ(s,a'), a' ~ G(s)} Q(s, a)

  其中约束 |ξ(s,a)| ≤ Φ 保证了"改进"不超出数据支撑范围。

论文: "Off-Policy Deep Reinforcement Learning without Exploration"
      Scott Fujimoto, David Meger, Doina Precup (ICML 2019)

适配 MovieLens 的关键改造:
  - 原始 BCQ 用于 MuJoCo (连续动作空间, ~10维)
  - MovieLens action space: 6万部电影, 离散
  - 改造: 在电影嵌入空间做 BCQ (ACTION_DIM=64 维连续空间),
    推理时用最近邻把嵌入向量映射回实际电影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import (
    ACTION_DIM, STATE_DIM, HIDDEN_DIM, LATENT_DIM,
    PERTURBATION_PHI, N_ACTIONS_SAMPLE,
)


# ============================================================
# VAE Generator — 行为策略建模
# ============================================================
class VAEGenerator(nn.Module):
    """
    条件 VAE: p(a | s) 的生成模型

    输入:  state + action
    编码:  (s, a) → μ, log_σ²
    采样:  z ~ N(μ, σ)
    解码:  (s, z) → â (重建的 action)

    训练时:  最小化重建误差 + KL 散度
    推理时:  输入 s, 从先验 z ~ N(0,1) 采样,
            decode 得到候选 action

    面试话术:
      "VAE 在这里不是做生成, 是做约束。
       它学到了'历史上给定 state s, 哪些 action 出现过'的分布。
       Q 网络只能在这个分布里选 action,
       从而避免了 offline RL 的核心问题: distribution shift。"
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: (s, a) → μ, log_σ²
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (s, z) → â
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode(self, state: torch.Tensor, action: torch.Tensor):
        """编码 (s,a) → μ, log_σ²"""
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # 裁剪 logvar 防数值爆炸
        logvar = torch.clamp(logvar, -10, 5)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """重参数化技巧: z = μ + σ·ε, ε ~ N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, state: torch.Tensor, z: torch.Tensor):
        """解码 (s,z) → â"""
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        训练时的前向: 编码再解码, 返回重建 + 隐变量参数
        """
        mu, logvar = self.encode(state, action)
        z = self.reparameterize(mu, logvar)
        recon_action = self.decode(state, z)
        return recon_action, mu, logvar

    def sample(self, state: torch.Tensor, n_samples: int = N_ACTIONS_SAMPLE):
        """
        推理时采样候选动作: s → {a₁, a₂, ..., a_N}

        从先验 N(0,1) 采样 z, decode 得到候选动作集。
        这 N 个候选动作就是 Q 网络的"搜索空间"。
        """
        batch_size = state.shape[0]
        device = state.device

        # 对每个 state 采样 n_samples 个 z
        state_expanded = state.unsqueeze(1).expand(-1, n_samples, -1)
        state_expanded = state_expanded.reshape(batch_size * n_samples, -1)

        z = torch.randn(batch_size * n_samples, self.latent_dim, device=device)
        actions = self.decode(state_expanded, z)
        actions = actions.reshape(batch_size, n_samples, self.action_dim)

        return actions

    def vae_loss(self, action: torch.Tensor, recon_action: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor):
        """
        VAE 损失 = 重建误差 (MSE) + KL 散度
        """
        recon_loss = F.mse_loss(recon_action, action, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # KL 权重: β-VAE 风格, 防止 KL collapse
        beta = min(1.0, 0.001 + 0.999 * (mu.shape[0] / 10000))
        return recon_loss + beta * 0.01 * kl_loss, recon_loss, kl_loss


# ============================================================
# Perturbation Network — 有限改进
# ============================================================
class Perturbation(nn.Module):
    """
    扰动网络 ξ(s, a): 对 VAE 生成的候选动作做小幅改进

    输出被 clip 到 [-Φ, Φ], 保证扰动幅度可控。
    Φ 越小 → 越保守 (更依赖 VAE), Φ 越大 → 越激进。

    面试话术:
      "Perturbation 是 BCQ 的精妙之处。
       纯 VAE 只能复现历史行为, 永远无法超越行为策略。
       Perturbation 允许在数据支撑区域内做小幅改进,
       幅度被 Φ 严格控制——这就是'有限探索'。"
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        phi: float = PERTURBATION_PHI,
    ):
        super().__init__()
        self.phi = phi

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # 初始化最后一层接近 0, 让初始扰动很小
        nn.init.uniform_(list(self.net.modules())[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(list(self.net.modules())[-1].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state:  (batch, state_dim)
            action: (batch, action_dim) or (batch, n_samples, action_dim)
        Returns:
            perturbed_action: same shape as action, clipped within [action - Φ, action + Φ]
        """
        # 处理多维 action (n_samples 维度)
        if action.dim() == 3:
            batch, n_samples, _ = action.shape
            state_expanded = state.unsqueeze(1).expand(-1, n_samples, -1)
            x = torch.cat([state_expanded, action], dim=-1)
            perturbation = self.net(x)
            perturbed = action + self.phi * torch.tanh(perturbation)
        else:
            x = torch.cat([state, action], dim=-1)
            perturbation = self.net(x)
            perturbed = action + self.phi * torch.tanh(perturbation)

        return perturbed


# ============================================================
# Q-Network — 动作价值评估
# ============================================================
class QNetwork(nn.Module):
    """
    Q(s, a): 评估在状态 s 下推荐电影 a 的长期期望收益

    用 Double Q-Learning: 两个独立的 Q 网络, 取较小值防过估计。

    面试话术:
      "Double DQN 用两个网络: 一个选 action, 另一个评 action。
      这解决了标准 DQN 的过估计 (overestimation) 问题——
      max 操作天然倾向于选 Q 值被高估的 action。"
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state:  (batch, state_dim) or (batch, n_samples, state_dim)
            action: match state shape
        Returns:
            q_value: (batch, 1) or (batch, n_samples, 1)
        """
        if action.dim() == 3 and state.dim() == 2:
            # state: (B, D), action: (B, N, D)
            state = state.unsqueeze(1).expand_as(action)

        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ============================================================
# BCQ 完整模型
# ============================================================
class BCQ(nn.Module):
    """
    BCQ 完整模型: VAE Generator + Perturbation + Double Q-Network

    用法:
      model = BCQ(state_dim, action_dim)
      loss = model.compute_loss(batch)
      loss.backward()
      optimizer.step()
      model.soft_update_target()
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        phi: float = PERTURBATION_PHI,
        gamma: float = 0.99,
        tau: float = 0.005,
        use_double_q: bool = True,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.use_double_q = use_double_q

        # 主网络
        self.generator = VAEGenerator(state_dim, action_dim, hidden_dim, latent_dim)
        self.perturbation = Perturbation(state_dim, action_dim, hidden_dim, phi)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)

        # Double Q (目标网络)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim)

        # 初始化目标网络权重 = 主网络权重
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def select_action(self, state: torch.Tensor, n_samples: int = None):
        """
        BCQ 策略: π(s) = argmax Q(s, a' + ξ(s, a'))  where a' ~ G(s)

        推理时的 action 选择:
          1. VAE 采样 N 个候选
          2. Perturbation 微调
          3. Q 网络打分
          4. 选最高分的

        Returns:
          best_action: (batch, action_dim)
          best_q:      (batch, 1)
        """
        if n_samples is None:
            n_samples = N_ACTIONS_SAMPLE

        with torch.no_grad():
            # Step 1: VAE 采样
            candidates = self.generator.sample(state, n_samples)  # (B, N, D)

            # Step 2: Perturbation
            candidates = self.perturbation(state, candidates)  # (B, N, D)

            # Step 3: Q 评估 (用 Q1)
            q_values = self.q1(state, candidates).squeeze(-1)  # (B, N)

            # Step 4: 选最佳
            best_idx = q_values.argmax(dim=1)  # (B,)
            best_action = candidates[torch.arange(candidates.shape[0]), best_idx]
            best_q = q_values[torch.arange(q_values.shape[0]), best_idx]

        return best_action, best_q.unsqueeze(-1)

    def compute_target(
        self,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        n_samples: int = None,
    ):
        """
        计算 Bellman 目标: y = r + γ * max_a' Q'(s', a')

        BCQ 的目标是用 VAE + Perturbation + Q_target 选最佳 next action:
          a' ~ G(s')  →  perturb →  Q_target(s', a')
        """
        if n_samples is None:
            n_samples = min(N_ACTIONS_SAMPLE, 50)  # 目标网络省点算力

        with torch.no_grad():
            # VAE 采样
            candidates = self.generator.sample(next_state, n_samples)
            # Perturbation
            candidates = self.perturbation(next_state, candidates)

            # Q 目标值
            q1_target_vals = self.q1_target(next_state, candidates).squeeze(-1)
            q2_target_vals = self.q2_target(next_state, candidates).squeeze(-1)

            if self.use_double_q:
                # Double Q: 用 Q1 选, Q2 评 (取 min)
                best_idx = q1_target_vals.argmax(dim=1)
                q_target = q2_target_vals[torch.arange(q2_target_vals.shape[0]), best_idx]
            else:
                # 标准: max Q_target
                q_target = torch.max(
                    torch.min(q1_target_vals, q2_target_vals), dim=1
                ).values

            q_target = q_target.unsqueeze(-1)
            target = reward + self.gamma * (1 - done) * q_target

        return target

    def compute_loss(self, batch: dict, vae_weight: float = 0.5):
        """
        一次完整的 BCQ 训练步骤的损失计算

        batch 包含:
          state:       (B, state_dim)
          action:      (B, action_dim)   当前 action 的电影嵌入
          reward:      (B, 1)
          next_state:  (B, state_dim)
          done:        (B, 1)

        返回:
          total_loss, loss_dict
        """
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        # --- 1. Q-Learning Loss ---
        target = self.compute_target(next_state, reward, done)
        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)
        q_loss = q1_loss + q2_loss

        # --- 2. VAE Loss ---
        recon_action, mu, logvar = self.generator(state, action)
        vae_total_loss, recon_loss, kl_loss = self.generator.vae_loss(
            action, recon_action, mu, logvar
        )
        vae_loss = vae_total_loss

        # --- 3. Perturbation Loss ---
        # 最大化 Q(s, a + ξ(s, a))
        perturbed_action = self.perturbation(state, action)
        q_perturbed = self.q1(state, perturbed_action)
        perturb_loss = -q_perturbed.mean()

        # --- 总损失 ---
        total_loss = q_loss + vae_weight * vae_loss + perturb_loss

        loss_dict = {
            "total": total_loss.item(),
            "q_loss": q_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "vae_loss": vae_loss.item(),
            "vae_recon": recon_loss.item(),
            "vae_kl": kl_loss.item(),
            "perturb_loss": perturb_loss.item(),
            "q1_mean": q1_pred.mean().item(),
            "target_mean": target.mean().item(),
        }

        return total_loss, loss_dict

    def soft_update_target(self):
        """软更新目标网络: θ' ← τ·θ + (1-τ)·θ'"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
