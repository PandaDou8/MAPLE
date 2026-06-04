import gc

import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
import logging
import os

class DistMultModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(DistMultModule, self).__init__()
        sigma = 0.2
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.rel_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.ent_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))
        # self.margin = 1.0  # 用于对齐性损失的 margin
        # self.t = 1.0  # 用于均匀性损失的温度参数

    def forward(self, src, rel, dst):

        return t.sum(self.ent_embed(dst) * self.ent_embed(src) * self.rel_embed(rel), dim=-1)

    def score(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def dist(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return self.forward(src, rel, dst)


    # def alignment_loss(self, src, rel, dst, alpha=2):
    #     """
    #     对齐性损失函数：最小化正样本对之间的距离。
    #     """
    #     d_good = self.dist(src, rel, dst)
    #     return t.mean(d_good ** alpha)

    # def uniformity_loss(self, src, rel, dst):
    #     """
    #     均匀性损失函数：最大化嵌入在单位超球面上的均匀性。
    #     """
    #     embeddings = self.ent_embed(dst)
    #     pairwise_dist = t.cdist(embeddings, embeddings, p=2)
    #     return t.log(t.mean(t.exp(-self.t * pairwise_dist ** 2)))

class DistMult(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(DistMult, self).__init__()
        self.mdl = DistMultModule(n_ent, n_rel, config)
        self.mdl.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester):
        src, dst, rel = train_data
        n_train = len(src)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        optimizer = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        best_perf = 0
        for epoch in range(n_epoch):
            epoch_loss = 0
            if epoch % self.config.sample_freq == 0:
                rand_idx = t.randperm(n_train)
                src = src[rand_idx]
                rel = rel[rand_idx]
                dst = dst[rand_idx]
                src_corrupted, rel_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
                src_corrupted = src_corrupted.cuda()
                rel_corrupted = rel_corrupted.cuda()
                dst_corrupted = dst_corrupted.cuda()
            for ss, rs, ts in batch_by_num(n_batch, src_corrupted, rel_corrupted, dst_corrupted, n_sample=n_train):
                self.mdl.zero_grad()
                label = t.zeros(len(ss)).type(t.LongTensor).cuda()
                loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), label))
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.data[0]
                #### 改 如果你有一个0维张量（即一个单独的数字），你不能使用tensor.data[0]来访问它，而应该使用tensor.item()
                epoch_loss += loss.data.item()
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, epoch_loss / n_train)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                test_perf = tester()
                if test_perf > best_perf:
                    # str = config().pretrain_dir + config().dataset["class"] + "/" + config().pretrain_config + ".mdl"
                    str = config().pretrain_gen_model
                    self.save(str)
                    # self.save(os.path.join(config().task.dir, self.config.model_file))
                    best_perf = test_perf
        return best_perf


    # def pretrain(self, train_data, corrupter, tester):
    #     src, dst, rel = train_data
    #     n_train = len(src)
    #     n_epoch = self.config.n_epoch
    #     n_batch = self.config.n_batch
    #     optimizer = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
    #     best_perf = 0

    #     for epoch in range(n_epoch):
    #         epoch_loss = 0
    #         if epoch % self.config.sample_freq == 0:
    #             rand_idx = t.randperm(n_train)
    #             src = src[rand_idx]
    #             rel = rel[rand_idx]
    #             dst = dst[rand_idx]
    #             src_corrupted, rel_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
    #             src_corrupted = src_corrupted.cuda()
    #             rel_corrupted = rel_corrupted.cuda()
    #             dst_corrupted = dst_corrupted.cuda()

    #         for ss, rs, ts in batch_by_num(n_batch, src_corrupted, rel_corrupted, dst_corrupted, n_sample=n_train):
    #             self.mdl.zero_grad()
    #             label = t.zeros(len(ss)).type(t.LongTensor).cuda()

    #             # 计算 Softmax 损失
    #             softmax_loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), label))

    #             # 计算对齐性损失
    #             align_loss = self.mdl.alignment_loss(Variable(ss), Variable(rs), Variable(ts))

    #             # 计算均匀性损失
    #             uniform_loss = self.mdl.uniformity_loss(Variable(ss), Variable(rs), Variable(ts))

    #             # 总损失 = Softmax 损失 + 对齐性损失 + 均匀性损失
    #             # total_loss = 0.2 * softmax_loss + 0.2 * align_loss + 0.6 * uniform_loss
    #             total_loss = 0.5 * align_loss + 0.5 * uniform_loss
    #             total_loss.backward()
    #             optimizer.step()
    #             epoch_loss += total_loss.data.item()

    #         logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, epoch_loss / n_train)

    #         if (epoch + 1) % self.config.epoch_per_test == 0:
    #             test_perf = tester()
    #             if test_perf > best_perf:
    #                 self.save(config().pretrain_gen_model)
    #                 best_perf = test_perf

    #     return best_perf