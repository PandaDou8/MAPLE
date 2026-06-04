import gc

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, src, rel, dst):
        raise NotImplementedError

    def dist(self, src, rel, dst):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst):
        raise NotImplementedError

    def prob(self, src, rel, dst):
        return nnf.softmax(self.prob_logit(src, rel, dst), dim=-1)

    def constraint(self):
        pass

    def pair_loss(self, src, rel, dst, src_bad, dst_bad):
        d_good = self.dist(src, rel, dst)
        d_bad = self.dist(src_bad, rel, dst_bad)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, truth):
        probs = self.prob(src, rel, dst)
        n = probs.size(0)
        device = probs.device
        truth_probs = torch.log(probs[torch.arange(0, n, device=device).long(), truth.to(device)] + 1e-30)
        return -truth_probs


class BaseModel(object):
    def __init__(self):
        self.mdl = None # type: BaseModule
        self.weight_decay = 0
        self.last_sample_info = None
        self.last_update_info = None
        self._optimizer_lr = None
        self._optimizer_betas = None

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        device = next(self.mdl.parameters()).device
        state = torch.load(filename, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.mdl.load_state_dict(state)

    def save_training_state(self, filename, extra=None):
        state = {
            "model": self.mdl.state_dict(),
            "optimizer": self.opt.state_dict() if hasattr(self, "opt") else None,
            "extra": extra or {},
        }
        torch.save(state, filename)

    def load_training_state(self, filename):
        device = next(self.mdl.parameters()).device
        state = torch.load(filename, map_location=device)
        self.mdl.load_state_dict(state["model"])
        self._ensure_optimizer()
        if state.get("optimizer") is not None:
            self.opt.load_state_dict(state["optimizer"])
            for opt_state in self.opt.state.values():
                for key, value in opt_state.items():
                    if isinstance(value, torch.Tensor):
                        opt_state[key] = value.to(device)
        return state.get("extra", {})

    def _trainable_parameters(self):
        return [param for param in self.mdl.parameters() if param.requires_grad]

    def _cfg_get(self, key, default):
        if not hasattr(self, "config"):
            return default
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _get_sample_cap(self, requested, available):
        cap = int(self._cfg_get("gan_sample_cap", 131072))
        return min(int(cap), int(requested), int(available))

    def _policy_gradient_enabled(self):
        return bool(self._cfg_get("enable_policy_gradient", True))

    def _get_generator_lr(self):
        if self._optimizer_lr is not None:
            return self._optimizer_lr
        self._optimizer_lr = float(self._cfg_get("policy_lr", self._cfg_get("generator_lr", 1.0e-4)))
        return self._optimizer_lr

    def _get_generator_betas(self):
        if self._optimizer_betas is not None:
            return self._optimizer_betas
        beta1 = float(self._cfg_get("policy_beta1", 0.9))
        beta2 = float(self._cfg_get("policy_beta2", 0.999))
        self._optimizer_betas = (beta1, beta2)
        return self._optimizer_betas

    def _get_grad_clip_norm(self):
        return float(self._cfg_get("generator_grad_clip_norm", 5.0))

    def _get_logit_clamp(self):
        return float(self._cfg_get("generator_logit_clamp", 50.0))

    def _ensure_optimizer(self):
        if hasattr(self, 'opt'):
            return
        trainable_params = self._trainable_parameters()
        if not trainable_params:
            raise ValueError("Generator has no trainable parameters")
        self.opt = Adam(
            trainable_params,
            lr=self._get_generator_lr(),
            betas=self._get_generator_betas(),
            weight_decay=self.weight_decay,
        )

    def _stable_softmax(self, logits):
        logit_clamp = self._get_logit_clamp()
        centered_logits = logits - logits.detach().max(dim=-1, keepdim=True).values
        if logit_clamp > 0:
            centered_logits = centered_logits.clamp(min=-logit_clamp, max=logit_clamp)
        probs = nnf.softmax(centered_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(dim=-1, keepdim=True)
        invalid_mask = row_sum.squeeze(-1) <= 0
        if invalid_mask.any():
            probs[invalid_mask] = 1.0 / probs.size(-1)
            row_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / row_sum.clamp_min(1e-12)
        return centered_logits, probs, invalid_mask

    def gen_samples(self, src, rel, dst, n_sample=1, temperature=1.0, train=True):
        n, m = dst.size()
        device = next(self.mdl.parameters()).device
        rel_var = Variable(rel.to(device))
        src_var = Variable(src.to(device))
        dst_var = Variable(dst.to(device))
        raw_logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / max(float(temperature), 1e-6)
        logits, probs, invalid_mask = self._stable_softmax(raw_logits)
        sample_size = min(int(n_sample), int(m))
        row_idx = torch.arange(0, n, device=device).long().unsqueeze(1).expand(n, sample_size)
        sample_idx = torch.multinomial(probs, sample_size, replacement=False)
        sample_srcs = src_var[row_idx, sample_idx]
        sample_dsts = dst_var[row_idx, sample_idx]
        self.last_sample_info = {
            "entropy": -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).detach(),
            "sample_probs": probs[row_idx, sample_idx].detach(),
            "sample_size": int(sample_size),
            "candidate_size": int(m),
            "requested_n_sample": int(n_sample),
            "temperature": float(temperature),
            "invalid_prob_rows": int(invalid_mask.sum().item()),
            "invalid_prob_row_ratio": float(invalid_mask.float().mean().item()),
            "logit_min": float(logits.min().item()),
            "logit_max": float(logits.max().item()),
            "logit_mean": float(logits.mean().item()),
            "logit_std": float(logits.std(unbiased=False).item()),
        }
        return sample_srcs, sample_dsts



    def gen_step(self, src, rel, dst, n_sample=200000, temperature=1.0, train=True):
        self._ensure_optimizer()
        n, m = dst.size()
        device = next(self.mdl.parameters()).device
        rel_var = Variable(rel.to(device))
        src_var = Variable(src.to(device))
        dst_var = Variable(dst.to(device))
        raw_logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / max(float(temperature), 1e-6)
        logits, probs, invalid_mask = self._stable_softmax(raw_logits)
        sample_size = self._get_sample_cap(n_sample, m)
        row_idx = torch.arange(0, n, device=device).long().unsqueeze(1).expand(n, sample_size)
        sample_idx = torch.multinomial(probs, sample_size, replacement=False)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        sample_probs = probs[row_idx, sample_idx].detach()
        self.last_sample_info = {
            "entropy": entropy.detach(),
            "sample_probs": sample_probs,
            "sample_size": int(sample_size),
            "candidate_size": int(m),
            "requested_n_sample": int(n_sample),
            "temperature": float(temperature),
            "invalid_prob_rows": int(invalid_mask.sum().item()),
            "invalid_prob_row_ratio": float(invalid_mask.float().mean().item()),
            "logit_min": float(logits.min().item()),
            "logit_max": float(logits.max().item()),
            "logit_mean": float(logits.mean().item()),
            "logit_std": float(logits.std(unbiased=False).item()),
        }
        sample_srcs = src_var[row_idx, sample_idx]
        sample_dsts = dst_var[row_idx, sample_idx]
        rewards = yield sample_srcs, sample_dsts
        self.last_update_info = {
            "reinforce_loss": 0.0,
            "generator_grad_norm": 0.0,
            "policy_update_enabled": float(self._policy_gradient_enabled()),
        }
        if train and self._policy_gradient_enabled() and rewards is not None:
            ### ===== 0424 FIX: rewards 必须和选中的 log_prob 一一对齐，形状必须是 [batch, sample] =====
            ### 如果上游误传成 [batch, 1, sample]，这里会发生广播并把不同 query 的梯度混在一起。
            if rewards.dim() != 2:
                raise ValueError("Generator rewards must be 2D [batch, sample], but got shape %s" % (tuple(rewards.shape),))
            if rewards.shape != sample_idx.shape:
                raise ValueError("Generator rewards shape %s doesn't match sampled index shape %s" %
                                 (tuple(rewards.shape), tuple(sample_idx.shape)))
            self.opt.zero_grad(set_to_none=True)
            log_probs = nnf.log_softmax(logits, dim=-1)
            rewards = rewards.to(device)
            selected_log_probs = log_probs[row_idx, sample_idx]
            reinforce_loss = -(rewards.detach() * selected_log_probs).mean()
            regularization_loss = torch.zeros((), device=device)
            if hasattr(self, "generator_regularization_loss"):
                regularization_loss = self.generator_regularization_loss()
            total_loss = reinforce_loss + regularization_loss
            total_loss.backward()
            grad_clip_norm = self._get_grad_clip_norm()
            grad_norm = 0.0
            if grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), grad_clip_norm).item())
            else:
                grad_sq = 0.0
                for param in self._trainable_parameters():
                    if param.grad is not None:
                        grad_sq += float(param.grad.detach().pow(2).sum().item())
                grad_norm = grad_sq ** 0.5
            self.opt.step()
            self.mdl.constraint()
            self.last_update_info = {
                "reinforce_loss": float(reinforce_loss.detach().item()),
                "regularization_loss": float(regularization_loss.detach().item()),
                "generator_grad_norm": float(grad_norm),
                "policy_update_enabled": 1.0,
            }
        yield None

    
    ## 正常采样
    # def gen_step(self, src, rel, dst, n_sample=1, temperature=1.0, train=True):
    #     if not hasattr(self, 'opt'):
    #         self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
    #     n, m = dst.size()
    #     rel_var = Variable(rel.cuda())
    #     src_var = Variable(src.cuda())
    #     dst_var = Variable(dst.cuda())
    #     logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
    #     probs = nnf.softmax(logits, dim=-1)
    #     row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
    #     sample_idx = torch.multinomial(probs, n_sample, replacement=False)
    #     sample_srcs = src[row_idx, sample_idx.data.cpu()]
    #     sample_dsts = dst[row_idx, sample_idx.data.cpu()]
    #     rewards = yield sample_srcs, sample_dsts
    #     if train:
    #         self.mdl.zero_grad()
    #         log_probs = nnf.log_softmax(logits, dim=-1)
    #         reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
    #         reinforce_loss.backward()
    #         self.opt.step()
    #         self.mdl.constraint()
    #     yield None


    def dis_step(self, src, rel, dst, src_fake, dst_fake, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src.cuda())
        rel_var = Variable(rel.cuda())
        dst_var = Variable(dst.cuda())
        src_fake_var = Variable(src_fake.cuda())
        dst_fake_var = Variable(dst_fake.cuda())
        losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var)
        fake_scores = self.mdl.score(src_fake_var, rel_var, dst_fake_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails, filt=True):

        mrr_tot = 0
        mr_tot = 0
        hit10_tot = 0
        count = 0
        for batch_s, batch_t, batch_r in batch_by_size(config().test_batch_size, *test_data):
            batch_size = batch_s.size(0)
            rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
            src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
            dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
            # all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
            #                    .type(torch.LongTensor).cuda(), volatile=True)
            ### 改 ：volatile已弃用
            with torch.no_grad():
                all_var = torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).type(torch.LongTensor).cuda()

                # all_var = torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                # all_var = all_var.type(torch.LongTensor).cuda()

            batch_dst_scores = self.mdl.score(src_var, rel_var, all_var).data
            batch_src_scores = self.mdl.score(all_var, rel_var, dst_var).data
            for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                # if filt:
                #     if tails[(s, r)]._nnz() > 1:
                #         tmp = dst_scores[t]
                #         dst_scores += tails[(s, r)].cuda() * 1e30
                #         dst_scores[t] = tmp
                #     if heads[(t, r)]._nnz() > 1:
                #         tmp = src_scores[s]
                #         src_scores += heads[(t, r)].cuda() * 1e30
                #         src_scores[s] = tmp
                # ## 改
                # if filt:
                #     if tails[(s.item(), r.item())]._nnz() > 1:
                #         tmp = dst_scores[t]
                #         dst_scores += tails[(s.item(), r.item())].cuda() * 1e30
                #         dst_scores[t] = tmp
                #     if heads[(t.item(), r.item())]._nnz() > 1:
                #         tmp = src_scores[s]
                #         src_scores += heads[(t.item(), r.item())].cuda() * 1e30
                #         src_scores[s] = tmp
                if filt:
                    if tails[(s.item(), r.item())]._nnz() > 1:
                        tmp = dst_scores[t].item()
                        dst_scores += tails[(s.item(), r.item())].cuda() * 1e30
                        dst_scores[t] = tmp
                    if heads[(t.item(), r.item())]._nnz() > 1:
                        tmp = src_scores[s].item()
                        src_scores += heads[(t.item(), r.item())].cuda() * 1e30
                        src_scores[s] = tmp

                mrr, mr, hit10 = mrr_mr_hitk(dst_scores, t)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                mrr, mr, hit10 = mrr_mr_hitk(src_scores, s)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit10_tot / count)
        return mrr_tot / count
