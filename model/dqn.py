"""
DQN (Deep Q-Network) — BCQ 的对比算法
================================================================
SparRL Phase 2.5 — 南志锦 · 2026-06-09

为什么需要 DQN 作为对比?

  BCQ 的核心卖点是"VAE 约束防止 distribution shift"。
  但如果面试官问"没有 VAE 会怎么样?"——你需要数据来证明。

  DQN 跟 BCQ 用相同的 Q 网络架构, 但去掉了 VAE 约束。
  在在线模拟中对比两者的学习曲线, 可以直观展示:
    - BCQ:  保守但稳定, NDCG 随数据增长稳中有升
    - DQN:  Q 值虚高, 早期 overestimation 导致策略差于 BCQ

  面试话术:
    "我做了 BCQ 和 DQN 的在线模拟对比。
     DQN 在数据少的早期轮次表现明显差于 BCQ,
     因为 Q 网络对 low-data 区域的 action 给出了虚高估值。
     BCQ 的 VAE 约束让策略只选'历史上有证据'的电影,
     数据越少, BCQ 的优势越明显。这就是 Distribution Shift 的实证。"

架构:
  - 复用 bcq.py 的 QNetwork (Double Q + target network)
  - ε-greedy 探索: 在候选动作集上做随机 vs 贪婪选择
  - Replay Buffer: 存储轨迹供采样训练
  - 候选动作: 从热门电影嵌入中随机采样 + Q 网络打分选择

设计原则 (跟 BCQ 公平对比):
  - 相同的 State Encoder
  - 相同的 Movie Embedding Lookup
  - 相同的 Q Network 结构 (Double Q)
  - 唯一的区别: BCQ 用 VAE 约束候选空间, DQN 自由探索
"""

import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (
    ACTION_DIM, STATE_DIM, HIDDEN_DIM,
    BATCH_SIZE, LEARNING_RATE, GAMMA, TAU, GRAD_CLIP, USE_DOUBLE_Q,
)


class ReplayBuffer:
    """
    经验回放缓冲区

    面试话术:
      "Experience Replay 是 DQN 的两大创新之一 (另一个是 Target Network)。
       它打破了序列数据的相关性, 让训练样本近似 i.i.d.,
       这是 Q-Learning 收敛的前提。

       优先级经验回放 (PER) 是后续改进——按 TD 误差的绝对值加权采样,
       让'学得不好的样本'被更频繁地复用。这里先用 uniform sampling
       保持简单, 面试时如果被问到可以说 PER 是下一步优化。"
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.cpu(), action.cpu(), reward.cpu(),
            next_state.cpu(), done.cpu(),
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "state": torch.stack(states).to(device),
            "action": torch.stack(actions).to(device),
            "reward": torch.stack(rewards).to(device),
            "next_state": torch.stack(next_states).to(device),
            "done": torch.stack(dones).to(device),
        }

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    DQN 模型: Double Q-Learning with ε-greedy exploration

    跟 BCQ 的区别: 不需要 VAE Generator 和 Perturbation。
    Action 选择是 ε-greedy 在候选电影嵌入上做 Q 值打分。

    面试话术 — 为什么 DQN 在离线场景会出问题:
      "DQN 的 max Q(s', a') 操作在 offline setting 下是致命的。
       因为 Q 网络会对那些'历史上很少或从未出现过的 action'
       给出不准确的估值, 而 max 操作天然倾向于选到这些被高估的 action。
       这就像你用有限的训练数据训练了一个回归模型,
       然后在未见过的区域做外推——外推值不可靠。
       BCQ 的 VAE 就是给这个 max 操作加了一个'安全带'。"
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        gamma: float = GAMMA,
        tau: float = TAU,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        use_double_q: bool = USE_DOUBLE_Q,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.use_double_q = use_double_q

        # ε-greedy 参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q 网络 (复用 BCQ 的 QNetwork)
        from model.bcq import QNetwork
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim)

        # 初始化目标网络
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def select_action(
        self,
        state: torch.Tensor,
        candidate_actions: torch.Tensor,  # (n_candidates, action_dim)
        evaluate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ε-greedy 动作选择。

        Args:
            state: (batch, state_dim)
            candidate_actions: (n_candidates, action_dim) 候选电影嵌入
            evaluate: True=纯贪婪 (评估模式), False=ε-greedy (训练模式)

        Returns:
            best_action: (batch, action_dim)
            best_q: (batch, 1)

        面试话术:
          "ε-greedy 是 exploration-exploitation 的最简单形式。
           训练时以概率 ε 随机选, 以概率 1-ε 选 Q 值最高的。
           ε 随时间衰减——一开始多探索, 后期多用学到的最优策略。

           在推荐场景, ε-greedy 的随机探索相当于'偶尔推一部
           用户可能没看过的电影', 这是冷启动和发现新兴趣的基础。"
        """
        batch_size = state.shape[0]
        n_candidates = candidate_actions.shape[0]
        device = state.device

        with torch.no_grad():
            # 扩展 state 以匹配候选动作
            state_expanded = state.unsqueeze(1).expand(-1, n_candidates, -1)
            actions_expanded = candidate_actions.unsqueeze(0).expand(batch_size, -1, -1)

            # Q 值评估 (用 Q1)
            q_values = self.q1(state_expanded, actions_expanded).squeeze(-1)  # (B, N)

            if evaluate or self.epsilon < self.epsilon_end:
                # 纯贪婪
                best_idx = q_values.argmax(dim=1)
            else:
                # ε-greedy: mask 掉随机选的不会的动作? 这里直接随机 vs 贪婪
                greedy_idx = q_values.argmax(dim=1)
                random_idx = torch.randint(0, n_candidates, (batch_size,), device=device)
                mask = torch.rand(batch_size, device=device) < self.epsilon
                best_idx = torch.where(mask, random_idx, greedy_idx)

            best_action = candidate_actions[best_idx]
            best_q = q_values[torch.arange(batch_size), best_idx]

        return best_action, best_q.unsqueeze(-1)

    def decay_epsilon(self):
        """衰减 ε"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def compute_target(
        self,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        candidate_actions: torch.Tensor,
    ):
        """
        计算 Bellman 目标: y = r + γ * max_a' Q_target(s', a')

        跟 BCQ 的关键区别:
          BCQ 的 max 只在 VAE 生成的候选里做 (constrained)
          DQN 的 max 在所有候选里做 (unconstrained)
          → DQN 更容易选中被高估的 action
        """
        batch_size = next_state.shape[0]
        n_candidates = candidate_actions.shape[0]
        device = next_state.device

        with torch.no_grad():
            # 扩展
            ns_expanded = next_state.unsqueeze(1).expand(-1, n_candidates, -1)
            ca_expanded = candidate_actions.unsqueeze(0).expand(batch_size, -1, -1)

            q1_target_vals = self.q1_target(ns_expanded, ca_expanded).squeeze(-1)
            q2_target_vals = self.q2_target(ns_expanded, ca_expanded).squeeze(-1)

            if self.use_double_q:
                best_idx = q1_target_vals.argmax(dim=1)
                q_target = q2_target_vals[torch.arange(batch_size), best_idx]
            else:
                q_target = torch.min(q1_target_vals, q2_target_vals).max(dim=1).values

            q_target = q_target.unsqueeze(-1)
            target = reward + self.gamma * (1 - done) * q_target

        return target

    def compute_loss(self, batch: dict, candidate_actions: torch.Tensor):
        """
        DQN 训练损失: TD 误差的 MSE

        跟 BCQ 的区别: 没有 VAE loss, 没有 perturbation loss。
        只有纯 Q-learning 目标。
        """
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        # TD target
        target = self.compute_target(next_state, reward, done, candidate_actions)

        # Q prediction
        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)

        # Loss
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        loss_dict = {
            "total": q1_loss.item() + q2_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q1_mean": q1_pred.mean().item(),
            "target_mean": target.mean().item(),
            "epsilon": self.epsilon,
        }

        return q1_loss + q2_loss, loss_dict

    def soft_update_target(self):
        """软更新目标网络"""
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
