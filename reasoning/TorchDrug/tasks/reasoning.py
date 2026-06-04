import torch
from torch import nn
import numpy as np

from torch.nn import functional as F
from torch.utils import data as torch_data

from reasoning.TorchDrug import core, tasks
from reasoning.TorchDrug.layers import functional
from reasoning.TorchDrug.core import Registry as R

@R.register("tasks.KnowledgeGraphCompletion")
class KnowledgeGraphCompletion(tasks.Task, core.Configurable):
    """
    Knowledge graph completion task.

    This class provides routines for the family of knowledge graph embedding models.

    Parameters:
        model (nn.Module): knowledge graph completion model
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``, ``ce`` and ``ranking``.
        metric (str or list of str, optional): metric(s). Available metrics are ``mr``, ``mrr`` and ``hits@K``.
        num_negative (int, optional): number of negative samples per positive sample
        margin (float, optional): margin in ranking criterion
        adversarial_temperature (float, optional): temperature for self-adversarial negative sampling.
            Set ``0`` to disable self-adversarial negative sampling.
        strict_negative (bool, optional): use strict negative sampling or not
        fact_ratio (float, optional): split the training set into facts and labels.
            Set ``None`` to use the whole training set as both facts and labels.
        sample_weight (bool, optional): whether to down-weight triplets from entities of large degrees
        filtered_ranking (bool, optional): use filtered or unfiltered ranking for evaluation
        full_batch_eval (bool, optional): whether to feed test negative samples by full batch or mini batch.
            Full batch speeds up evaluation significantly, but may cause OOM problems for some models and datasets.
    """
    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, fact_ratio=None,
                 sample_weight=True, filtered_ranking=True, full_batch_eval=False):
        super(KnowledgeGraphCompletion, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.num_negative = num_negative
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        self.strict_negative = strict_negative
        self.fact_ratio = fact_ratio
        self.sample_weight = sample_weight
        self.filtered_ranking = filtered_ranking
        self.full_batch_eval = full_batch_eval

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("graph", dataset.graph)
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        fact_mask[valid_set.indices] = 0
        fact_mask[test_set.indices] = 0
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
        self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        return train_set, valid_set, test_set

###  TODO reasoning.TorchDrug
    def forward(self, batch, all_loss=None, metric=None, is_gan=False,gen=None,point=0,src_smpl=None, dst_smpl = None, rel_smpl = None):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        ### 从这里切入，融合对抗训练
        if is_gan:
            pred = self.gan_predict(batch, all_loss, metric,point=point,gen=gen,src_smpl=src_smpl, dst_smpl = dst_smpl, rel_smpl = rel_smpl)
            pos_h_index, pos_t_index, pos_r_index = batch[0:3]
        else:
            # pred = self.predict(batch, all_loss, metric,point=point)
            pred = self.predict(batch, all_loss, metric, point=point)

            pos_h_index, pos_t_index, pos_r_index = batch.t()

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                neg_weight = torch.ones_like(pred)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / self.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / self.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            elif criterion == "ce":
                target = torch.zeros(len(pred), dtype=torch.long, device=self.device)
                loss = F.cross_entropy(pred, target, reduction="none")
            elif criterion == "ranking":
                positive = pred[:, :1]
                negative = pred[:, 1:]
                target = torch.ones_like(negative)
                loss = F.margin_ranking_loss(positive, negative, target, margin=self.margin)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            if self.sample_weight:
                sample_weight = self.degree_hr[pos_h_index, pos_r_index] * self.degree_tr[pos_t_index, pos_r_index]
                sample_weight = 1 / sample_weight.float().sqrt()
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            else:
                loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight
        ### TODO reasoning.TorchDrug 多返回一个pred

        # pred = F.softmax(pred, dim=-1)
        # return all_loss, metric, pred*(-1)
        return all_loss, metric, pred

    ### TODO reasoning.TorchDrug 使用生成器生成的负样本训练
    def gan_predict(self, batch, all_loss=None, metric=None,point=0,gen=None,src_smpl=None, dst_smpl = None, rel_smpl = None):

        pos_h_index, pos_t_index, pos_r_index = batch[0],batch[1],batch[2]
        batch_size = len(batch)
        if all_loss is None:
            # test
            all_index = torch.arange(self.num_entity, device=self.device)
            t_preds = []
            h_preds = []
            num_negative = self.num_entity if self.full_batch_eval else self.num_negative
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # print(" batch[5]")
            # print( batch[5])
            pred = self.model(self.fact_graph, src_smpl,dst_smpl,rel_smpl, all_loss=all_loss, metric=metric,point=point,gen=gen)
        return pred


    # ##### 负样本生成并保存
    # def predict(self, batch, all_loss=None, metric=None,point=0):
    #     pos_h_index, pos_t_index, pos_r_index = batch.t()
    #     batch_size = len(batch)

    #     if all_loss is None:
    #         # test
    #         all_index = torch.arange(self.num_entity, device=self.device)
    #         t_preds = []
    #         h_preds = []
    #         num_negative = self.num_entity if self.full_batch_eval else self.num_negative
    #         for neg_index in all_index.split(num_negative):
    #             r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
    #             h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
    #             t_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
    #             t_preds.append(t_pred)
    #         t_pred = torch.cat(t_preds, dim=-1)
    #         for neg_index in all_index.split(num_negative):
    #             r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
    #             t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
    #             h_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
    #             h_preds.append(h_pred)
    #         h_pred = torch.cat(h_preds, dim=-1)
    #         pred = torch.stack([t_pred, h_pred], dim=1)
    #         # in case of GPU OOM
    #         pred = pred.cpu()
    #     else:
    #         # train
            
    #         ### >=5时应该改回来2000
    #         if point < 0: 
    #             self.num_negative = 32
    #         else:
    #             self.num_negative = 2000
            
            
    #         if self.strict_negative:
    #             neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
    #         else:
    #             neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
                
    #         # neg_index就是候选样本
    #         gen_config = config()["DistMult"]
    #         # gen = DistMult(40943, 11, gen_config)
    #         gen = DistMult(14541, 237, gen_config)


    #         gen_model_path = config().pretrain_gen_model
    #         gen.load(gen_model_path)
            
    #         # print("-----point------")
    #         # print(point)
    #         if point >= 0:
    #             h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
    #             t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
    #             r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
    #             t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
    #             h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
    #             # print("======h_index========")
    #             # print(h_index.shape)
    #             # print(h_index)
    #             # print("==============")  
                
    #             src_smpl, dst_smpl = gen.gen_samples(h_index[:,1:], r_index[:,1:], t_index[:,1:], n_sample = 32, temperature=0.5)
    #             # print("=======mid=======")
    #             # print(src_smpl)
    #             # print(dst_smpl)
    #             # print("==============")     
    #             # print("==========666=========")
    #             # print(pos_h_index)
    #             # print(h_index.shape)
    #             # print(h_index)
    #             # print("===================")
                
    #             src_smpl = torch.cat((pos_h_index.unsqueeze(1), src_smpl), dim=1)
    #             dst_smpl = torch.cat((pos_t_index.unsqueeze(1), dst_smpl), dim=1)
    #             rel_smpl = r_index[:,:32 + 1]
    #             pred = self.model(self.fact_graph, src_smpl, dst_smpl, rel_smpl, all_loss=all_loss, metric=metric,point=point)
    #         else:
                
    #             h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
    #             t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
    #             r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
    #             t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
    #             h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
    #             pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric,point=point)
            
    #         # print(src_smpl.shape)
    #         # print(src_smpl)
            
            

    #         # pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

    #         # pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric,point=point)

    #     return pred

    ##### 负样本生成并保存
    def predict(self, batch, all_loss=None, metric=None,point=0):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)

        if all_loss is None:
            # test
            all_index = torch.arange(self.num_entity, device=self.device)
            t_preds = []
            h_preds = []
            num_negative = self.num_entity if self.full_batch_eval else self.num_negative
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # 0430 CLEANUP: 普通训练分支只保留干净的 strict-negative / random-negative 逻辑。
            # 这里不再每个 batch 重新实例化 / 加载旧 DistMult，也不再混入历史 hard-negative 筛样逻辑。
            # 当前仓库的强化学习 / 对抗训练统一走 engine.gan_train_astar()，避免 baseline 被旧代码污染。
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)

            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)

            split_point = batch_size // 2
            t_index[:split_point, 1:] = neg_index[:split_point]
            h_index[split_point:, 1:] = neg_index[split_point:]
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, point=point)
            
        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = self.graph.match(pattern)
        t_truth_index = self.graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = self.graph.match(pattern)
        h_truth_index = self.graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()

    def evaluate(self, pred, target):
        mask, target = target

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if self.filtered_ranking:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1

        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).float().mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric

    def visualize(self, batch):
        h_index, t_index, r_index = batch.t()
        return self.model.visualize(self.fact_graph, h_index, t_index, r_index)

    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index):
        batch_size = len(pos_h_index)
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        pattern = pattern[:batch_size // 2]
        edge_index, num_t_truth = self.fact_graph.match(pattern)
        t_truth_index = self.fact_graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        pattern = pattern[batch_size // 2:]
        edge_index, num_h_truth = self.fact_graph.match(pattern)
        h_truth_index = self.fact_graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

        neg_index = torch.cat([neg_t_index, neg_h_index])

        return neg_index
