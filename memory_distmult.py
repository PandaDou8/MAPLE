import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel, BaseModule


logger = logging.getLogger(__name__)


def _cfg_get(config, key, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class MemoryAugmentedDistMultModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(MemoryAugmentedDistMultModule, self).__init__()
        sigma = 0.2
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = int(config.dim)
        self.policy_hidden_dim = int(_cfg_get(config, "policy_hidden_dim", self.dim))
        self.memory_dim = int(_cfg_get(config, "memory_dim", max(16, self.dim // 4)))
        self.memory_momentum = float(_cfg_get(config, "memory_momentum", 0.95))
        self.memory_topk = int(_cfg_get(config, "memory_topk", 256))
        self.memory_reward_clip = float(_cfg_get(config, "memory_reward_clip", 10.0))
        self.policy_dropout = float(_cfg_get(config, "policy_dropout", 0.0))
        self.scale_clamp = float(_cfg_get(config, "generator_scale_clamp", 12.0))
        self.residual_score_clamp = float(_cfg_get(config, "generator_residual_score_clamp", 8.0))
        self.scale_regularization = float(_cfg_get(config, "generator_scale_regularization", 1.0e-4))

        # Keep the original DistMult parameter names so legacy checkpoints load directly.
        self.rel_embed = nn.Embedding(n_rel, self.dim)
        self.rel_embed.weight.data.div_((self.dim / sigma ** 2) ** (1 / 6))
        self.ent_embed = nn.Embedding(n_ent, self.dim)
        self.ent_embed.weight.data.div_((self.dim / sigma ** 2) ** (1 / 6))

        # Query-conditioned policy head. The last layer is zero-init so the initial
        # generator is still identical to the pretrained DistMult prior.
        self.policy_src = nn.Linear(self.dim, self.policy_hidden_dim, bias=False)
        self.policy_dst = nn.Linear(self.dim, self.policy_hidden_dim, bias=False)
        self.policy_pair = nn.Linear(self.dim, self.policy_hidden_dim, bias=False)
        self.policy_rel = nn.Linear(self.dim, self.policy_hidden_dim, bias=True)
        self.policy_norm = nn.LayerNorm(self.policy_hidden_dim)
        self.policy_out = nn.Linear(self.policy_hidden_dim, 1, bias=True)
        nn.init.zeros_(self.policy_out.weight)
        nn.init.zeros_(self.policy_out.bias)

        self.adapter_scale = nn.Parameter(
            torch.tensor(float(_cfg_get(config, "adapter_scale_init", 1.0)), dtype=torch.float32)
        )
        self.memory_scale = nn.Parameter(
            torch.tensor(float(_cfg_get(config, "memory_scale_init", 1.0)), dtype=torch.float32)
        )

        # Relation-aware episodic vector memory:
        # each entity stores a compact vector; the current relation + peer entity
        # defines a context vector, and scoring uses their similarity.
        self.head_context_rel = nn.Linear(self.dim, self.memory_dim, bias=False)
        self.head_context_peer = nn.Linear(self.dim, self.memory_dim, bias=True)
        self.tail_context_rel = nn.Linear(self.dim, self.memory_dim, bias=False)
        self.tail_context_peer = nn.Linear(self.dim, self.memory_dim, bias=True)

        self.register_buffer("head_memory_bank", torch.zeros(n_ent, self.memory_dim))
        self.register_buffer("tail_memory_bank", torch.zeros(n_ent, self.memory_dim))
        self.freeze_backbone()

    def freeze_backbone(self):
        self.ent_embed.weight.requires_grad_(False)
        self.rel_embed.weight.requires_grad_(False)

    def _policy_score(self, src_rel, dst_rel, pair_embed, rel_embed):
        hidden = self.policy_src(src_rel)
        hidden = hidden + self.policy_dst(dst_rel)
        hidden = hidden + self.policy_pair(pair_embed)
        hidden = hidden + self.policy_rel(rel_embed)
        hidden = self.policy_norm(hidden)
        hidden = torch.tanh(hidden)
        if self.policy_dropout > 0:
            hidden = F.dropout(hidden, p=self.policy_dropout, training=self.training)
        return self.policy_out(hidden).squeeze(-1)

    def _head_context(self, rel_embed, dst_embed):
        context = self.head_context_rel(rel_embed) + self.head_context_peer(dst_embed)
        return torch.tanh(context)

    def _tail_context(self, rel_embed, src_embed):
        context = self.tail_context_rel(rel_embed) + self.tail_context_peer(src_embed)
        return torch.tanh(context)

    def score_components(self, src, rel, dst):
        src_embed = self.ent_embed(src)
        rel_embed = self.rel_embed(rel)
        dst_embed = self.ent_embed(dst)

        src_rel = src_embed * rel_embed
        dst_rel = dst_embed * rel_embed
        pair_embed = src_embed * dst_embed

        base_score = torch.sum(src_rel * dst_embed, dim=-1)
        residual_score = self._policy_score(src_rel, dst_rel, pair_embed, rel_embed)
        if self.residual_score_clamp > 0:
            residual_score = residual_score.clamp(
                min=-self.residual_score_clamp,
                max=self.residual_score_clamp,
            )

        head_context = self._head_context(rel_embed, dst_embed)
        tail_context = self._tail_context(rel_embed, src_embed)
        head_memory_score = torch.sum(self.head_memory_bank[src] * head_context, dim=-1)
        tail_memory_score = torch.sum(self.tail_memory_bank[dst] * tail_context, dim=-1)
        memory_score = torch.tanh(head_memory_score + tail_memory_score)
        if self.scale_clamp > 0:
            effective_adapter_scale = self.adapter_scale.clamp(-self.scale_clamp, self.scale_clamp)
            effective_memory_scale = self.memory_scale.clamp(-self.scale_clamp, self.scale_clamp)
        else:
            effective_adapter_scale = self.adapter_scale
            effective_memory_scale = self.memory_scale

        full_score = base_score + effective_adapter_scale * residual_score + effective_memory_scale * memory_score
        return {
            "base": base_score,
            "residual": residual_score,
            "memory": memory_score,
            "full": full_score,
            "adapter_scale_effective": torch.full_like(base_score, float(effective_adapter_scale.detach().item())),
            "memory_scale_effective": torch.full_like(base_score, float(effective_memory_scale.detach().item())),
        }

    def forward(self, src, rel, dst):
        return self.score_components(src, rel, dst)["full"]

    def score(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def dist(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return self.forward(src, rel, dst)

    @torch.no_grad()
    def update_memory(self, src, rel, dst, reward):
        if reward is None or reward.numel() == 0:
            return

        reward = reward.detach().to(self.head_memory_bank.device, dtype=self.head_memory_bank.dtype)
        src = src.detach().to(self.head_memory_bank.device, dtype=torch.long)
        rel = rel.detach().to(self.head_memory_bank.device, dtype=torch.long)
        dst = dst.detach().to(self.head_memory_bank.device, dtype=torch.long)

        reward = reward.clamp(min=0, max=self.memory_reward_clip)
        if reward.dim() != 2:
            raise ValueError("Memory reward must be 2D [batch, sample], but got %s" % (tuple(reward.shape),))

        topk = reward.shape[1] if self.memory_topk <= 0 else min(self.memory_topk, reward.shape[1])
        if topk < reward.shape[1]:
            reward, index = reward.topk(topk, dim=-1)
            src = src.gather(1, index)
            rel = rel.gather(1, index)
            dst = dst.gather(1, index)

        positive_mask = reward > 0
        if not positive_mask.any():
            return

        src_embed = self.ent_embed(src)
        rel_embed = self.rel_embed(rel)
        dst_embed = self.ent_embed(dst)
        head_context = self._head_context(rel_embed, dst_embed)
        tail_context = self._tail_context(rel_embed, src_embed)

        self._update_bank(self.head_memory_bank, src[positive_mask], head_context[positive_mask], reward[positive_mask])
        self._update_bank(self.tail_memory_bank, dst[positive_mask], tail_context[positive_mask], reward[positive_mask])

    @torch.no_grad()
    def _update_bank(self, bank, entity, context, reward):
        context = F.normalize(context, dim=-1, eps=1e-6)
        unique_entity, inverse = torch.unique(entity, return_inverse=True)
        context_sum = torch.zeros(len(unique_entity), self.memory_dim, device=bank.device, dtype=bank.dtype)
        context_sum.index_add_(0, inverse, context * reward.unsqueeze(-1))
        reward_sum = torch.zeros(len(unique_entity), device=bank.device, dtype=bank.dtype)
        reward_sum.index_add_(0, inverse, reward)
        context_mean = context_sum / reward_sum.unsqueeze(-1).clamp_min(1e-6)
        bank[unique_entity] = (
            self.memory_momentum * bank[unique_entity]
            + (1 - self.memory_momentum) * context_mean
        )


class MemoryAugmentedDistMult(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(MemoryAugmentedDistMult, self).__init__()
        self.mdl = MemoryAugmentedDistMultModule(n_ent, n_rel, config)
        if torch.cuda.is_available():
            self.mdl.cuda()
        self.config = config
        self.weight_decay = _cfg_get(config, "lam", 0) / max(_cfg_get(config, "n_batch", 1), 1)

    def load(self, filename):
        device = next(self.mdl.parameters()).device
        state = torch.load(filename, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = self.mdl.load_state_dict(state, strict=False)
        self.mdl.freeze_backbone()

        if missing:
            logger.info("Generator checkpoint missing keys (expected for prior-initialized policy head): %s", missing)
        if unexpected:
            logger.info("Generator checkpoint has unexpected keys: %s", unexpected)

    @torch.no_grad()
    def update_memory(self, src, rel, dst, reward):
        self.mdl.update_memory(src, rel, dst, reward)

    @torch.no_grad()
    def score_components(self, src, rel, dst):
        return self.mdl.score_components(src, rel, dst)

    def generator_regularization_loss(self):
        reg = self.mdl.scale_regularization
        if reg <= 0:
            device = next(self.mdl.parameters()).device
            return torch.zeros((), device=device)
        return reg * (self.mdl.adapter_scale.pow(2) + self.mdl.memory_scale.pow(2))
