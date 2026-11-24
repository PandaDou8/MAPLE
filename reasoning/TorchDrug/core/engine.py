import os
import sys
import logging
import gc
import csv
import time
from itertools import islice
from collections import defaultdict

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
from torch.utils.data import TensorDataset

from reasoning.TorchDrug import data, core, utils
from reasoning.TorchDrug.core import Registry as R
from reasoning.TorchDrug.utils import comm, pretty
from corrupter import BernCorrupterMulti

module = sys.modules[__name__]
logger = logging.getLogger(__name__)


@R.register("core.Engine")
class Engine(core.Configurable):
    """
    General class that handles everything about training and test of a task.

    This class can perform synchronous distributed parallel training over multiple CPUs or GPUs.
    To invoke parallel training, launch with one of the following commands.

    1. Single-node multi-process case.

    .. code-block:: bash

        python -m torch.distributed.launch --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}

    2. Multi-node multi-process case.

    .. code-block:: bash

        python -m torch.distributed.launch --nnodes={number_of_nodes} --node_rank={rank_of_this_node}
        --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}

    If :meth:`preprocess` is defined by the task, it will be applied to ``train_set``, ``valid_set`` and ``test_set``.

    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multi-process case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        log_interval (int, optional): log every n gradient updates
    """

    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1,
                 gradient_interval=1, num_worker=0, logger="logging", log_interval=100,
                 trace_query_count=32, trace_topk=16, trace_every_n_epoch=1, trace_seed=1024,
                 policy_metric_sample_size=2048):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.trace_query_count = int(trace_query_count)
        self.trace_topk = int(trace_topk)
        self.trace_every_n_epoch = max(int(trace_every_n_epoch), 1)
        self.trace_seed = int(trace_seed)
        self.policy_metric_sample_size = max(int(policy_metric_sample_size), 0)
        self.trace_output_dir = os.path.abspath("negative_traces")
        self.train_epoch_metrics_file = os.path.abspath("train_epoch_metrics.csv")
        self.policy_epoch_metrics_file = os.path.abspath("policy_epoch_metrics.csv")
        self.structured_logging_enabled = self.rank == 0
        self._mediator_types = ("Chemical", "Metabolite", "Protein", "Reaction")
        self._neighbor_cache = None
        self._entity_vocab = None
        self._relation_vocab = None
        self._entity_types = None
        self._triplets = None
        self._train_indices = None
        self._trace_queries = []
        self._trace_query_map = {}
        self.gan_avg_reward = None
        self.gan_global_epoch = 0

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0, logger=logger)
        self.meter.log_config(self.config_dict())
        self._init_structured_logging()

    def _init_structured_logging(self):
        self._prepare_dataset_metadata()
        if not self.structured_logging_enabled:
            return

        self._ensure_csv_header(self.train_epoch_metrics_file, [
            "epoch", "discriminator_loss", "reward_mean", "reward_nonzero_ratio", "ema_baseline",
            "batch_size", "num_negative", "gan_sample_cap", "requested_n_sample", "effective_sample_size",
            "generator_temperature", "gpu_peak_mem_mb", "epoch_seconds",
            "policy_update_enabled", "generator_grad_norm", "discriminator_grad_norm", "reinforce_loss",
            "invalid_prob_row_count", "invalid_prob_row_ratio",
            "logit_min", "logit_max", "logit_mean", "logit_std"
        ])
        self._ensure_csv_header(self.policy_epoch_metrics_file, [
            "epoch", "component_probe_size", "base_score_mean", "base_score_std",
            "residual_score_mean", "residual_score_std", "memory_score_mean", "memory_score_std",
            "full_score_mean", "full_score_std", "policy_entropy_mean", "policy_entropy_std",
            "sample_prob_mean", "sample_prob_std", "reward_mean", "reward_std",
            "hardest_reward_mean", "hardest_reward_std", "hardest_reward_nonzero_ratio",
            "sampled_entity_unique_ratio", "memory_active_ratio", "adapter_scale", "memory_scale",
            "head_memory_norm", "tail_memory_norm", "adapter_scale_effective", "memory_scale_effective"
        ])
        if self.trace_query_count > 0:
            os.makedirs(self.trace_output_dir, exist_ok=True)

    def _prepare_dataset_metadata(self):
        dataset = self.train_set.dataset if isinstance(self.train_set, torch_data.Subset) else self.train_set
        indices = list(self.train_set.indices) if isinstance(self.train_set, torch_data.Subset) else list(range(len(self.train_set)))
        self._train_indices = indices
        self._triplets = getattr(dataset, "triplets", None)

        entity_vocab = getattr(dataset, "entity_vocab", None)
        relation_vocab = getattr(dataset, "relation_vocab", None)
        if entity_vocab is not None:
            self._entity_vocab = list(entity_vocab)
            self._entity_types = [self._entity_type(name) for name in self._entity_vocab]
        if relation_vocab is not None:
            self._relation_vocab = list(relation_vocab)

        if self.trace_query_count > 0 and self._triplets is not None and self._train_indices:
            self._build_trace_queries()

    def _build_trace_queries(self):
        generator = torch.Generator()
        generator.manual_seed(self.trace_seed)
        order = torch.randperm(len(self._train_indices), generator=generator).tolist()
        seen = set()
        self._trace_queries = []
        self._trace_query_map = {}
        for pos in order:
            triplet = self._triplets[self._train_indices[pos]]
            key = tuple(int(x) for x in triplet.tolist())
            if key in seen:
                continue
            seen.add(key)
            self._trace_query_map[key] = len(self._trace_queries)
            self._trace_queries.append(key)
            if len(self._trace_queries) >= self.trace_query_count:
                break

    @staticmethod
    def _ensure_csv_header(path, fieldnames):
        if os.path.exists(path):
            return
        with open(path, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

    @staticmethod
    def _append_csv_row(path, fieldnames, row):
        with open(path, "a", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writerow(row)

    @staticmethod
    def _entity_type(name):
        if "//" in name:
            return name.split("//", 1)[0]
        return name.split(":", 1)[0]

    @staticmethod
    def _relation_family(name):
        return name.split("_", 1)[0]

    def _entity_name(self, entity_id):
        if self._entity_vocab is None or entity_id >= len(self._entity_vocab):
            return str(entity_id)
        return self._entity_vocab[entity_id]

    def _relation_name(self, relation_id):
        if self._relation_vocab is None or relation_id >= len(self._relation_vocab):
            return str(relation_id)
        return self._relation_vocab[relation_id]

    def _entity_type_by_id(self, entity_id):
        if self._entity_types is None or entity_id >= len(self._entity_types):
            return "Unknown"
        return self._entity_types[entity_id]

    def _ensure_neighbor_cache(self):
        if self._neighbor_cache is not None:
            return
        if self._triplets is None or self._entity_types is None:
            self._neighbor_cache = {}
            return

        if self.structured_logging_enabled:
            logger.info("Building mediator neighbor cache for negative trace logging")

        neighbor_cache = defaultdict(lambda: defaultdict(set))
        for chunk in self._triplets.split(2_000_000):
            for src, dst, _ in chunk.tolist():
                dst_type = self._entity_types[dst]
                if dst_type in self._mediator_types:
                    neighbor_cache[int(src)][dst_type].add(int(dst))
                src_type = self._entity_types[src]
                if src_type in self._mediator_types:
                    neighbor_cache[int(dst)][src_type].add(int(src))
        self._neighbor_cache = neighbor_cache

    def _mediator_overlap(self, positive_entity, negative_entity):
        self._ensure_neighbor_cache()
        pos_neighbors = self._neighbor_cache.get(int(positive_entity), {})
        neg_neighbors = self._neighbor_cache.get(int(negative_entity), {})
        return {
            "%s_overlap" % mediator: len(pos_neighbors.get(mediator, set()) & neg_neighbors.get(mediator, set()))
            for mediator in self._mediator_types
        }

    @staticmethod
    def _cfg_get(config, key, default):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def _accumulate_tensor_stat(accumulator, name, tensor):
        if tensor is None or tensor.numel() == 0:
            return
        tensor = tensor.detach().float()
        accumulator["%s_sum" % name] += tensor.sum().item()
        accumulator["%s_sq_sum" % name] += (tensor * tensor).sum().item()
        accumulator["%s_count" % name] += tensor.numel()

    @staticmethod
    def _accumulate_scalar_mean(accumulator, name, value):
        accumulator["%s_sum" % name] += float(value)
        accumulator["%s_count" % name] += 1.0

    @staticmethod
    def _batch_size_of(batch):
        if isinstance(batch, torch.Tensor):
            return int(batch.size(0))
        if isinstance(batch, (list, tuple)) and batch:
            return int(batch[0].size(0))
        raise ValueError("Unsupported batch type `%s`" % type(batch))

    @staticmethod
    def _finalize_stat(accumulator, name):
        count = accumulator.get("%s_count" % name, 0.0)
        if count <= 0:
            return 0.0, 0.0
        mean = accumulator.get("%s_sum" % name, 0.0) / count
        mean_sq = accumulator.get("%s_sq_sum" % name, 0.0) / count
        var = max(mean_sq - mean * mean, 0.0)
        return mean, var ** 0.5

    @staticmethod
    def _model_grad_norm(parameters):
        sq_sum = 0.0
        found = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            sq_sum += grad.float().pow(2).sum().item()
            found = True
        if not found:
            return 0.0
        return sq_sum ** 0.5

    def _collect_trace_rows(self, epoch_id, batch_id, batch, neg_src_smpl, neg_dst_smpl, neg_rel_smpl,
                            pred, rewards, sample_probs, generator, trace_seen):
        if not self._trace_query_map:
            return []

        rows = []
        heads = batch[0].detach().cpu()
        tails = batch[1].detach().cpu()
        rels = batch[2].detach().cpu()
        neg_src_cpu = neg_src_smpl.detach().cpu()
        neg_dst_cpu = neg_dst_smpl.detach().cpu()
        neg_rel_cpu = neg_rel_smpl.detach().cpu()
        pred_cpu = pred.detach().cpu()
        rewards_cpu = rewards.detach().cpu()
        sample_probs_cpu = sample_probs.detach().cpu() if sample_probs is not None else None
        split_point = self.batch_size // 2

        for row_idx in range(heads.size(0)):
            key = (int(heads[row_idx]), int(tails[row_idx]), int(rels[row_idx]))
            if key not in self._trace_query_map or key in trace_seen:
                continue
            trace_seen.add(key)
            trace_query_idx = self._trace_query_map[key]
            is_head_corrupt = row_idx >= split_point
            anchor_entity = int(tails[row_idx]) if is_head_corrupt else int(heads[row_idx])
            positive_entity = int(heads[row_idx]) if is_head_corrupt else int(tails[row_idx])
            negative_entities = neg_src_cpu[row_idx] if is_head_corrupt else neg_dst_cpu[row_idx]

            with torch.no_grad():
                row_components = generator.score_components(
                    neg_src_smpl[row_idx: row_idx + 1],
                    neg_rel_smpl[row_idx: row_idx + 1],
                    neg_dst_smpl[row_idx: row_idx + 1]
                )
            row_components = {name: value[0].detach().cpu() for name, value in row_components.items()}
            full_order = torch.argsort(row_components["full"], descending=True)
            topk = min(self.trace_topk, full_order.numel())
            if topk <= 0:
                continue
            top_index = full_order[:topk]

            prior_order = torch.argsort(row_components["base"], descending=True)
            prior_rank = torch.empty_like(prior_order)
            prior_rank[prior_order] = torch.arange(1, prior_order.numel() + 1)

            current_neg_scores = pred_cpu[row_idx, 1:]
            current_pos = float(pred_cpu[row_idx, 0])
            current_hardest = int(torch.argmax(current_neg_scores).item()) if current_neg_scores.numel() else -1
            prior_hardest = int(torch.argmax(row_components["base"]).item()) if row_components["base"].numel() else -1
            relation_id = int(rels[row_idx])
            relation_name = self._relation_name(relation_id)

            for rank_pos, col_idx in enumerate(top_index.tolist(), start=1):
                negative_entity = int(negative_entities[col_idx])
                overlap = self._mediator_overlap(positive_entity, negative_entity)
                rows.append({
                    "epoch": int(epoch_id),
                    "batch_id": int(batch_id),
                    "trace_query_idx": int(trace_query_idx),
                    "source": "policy",
                    "corruption_side": "head" if is_head_corrupt else "tail",
                    "head_id": int(heads[row_idx]),
                    "tail_id": int(tails[row_idx]),
                    "relation_id": relation_id,
                    "head_name": self._entity_name(int(heads[row_idx])),
                    "tail_name": self._entity_name(int(tails[row_idx])),
                    "relation_name": relation_name,
                    "relation_family": self._relation_family(relation_name),
                    "anchor_entity": anchor_entity,
                    "anchor_name": self._entity_name(anchor_entity),
                    "anchor_type": self._entity_type_by_id(anchor_entity),
                    "positive_entity": positive_entity,
                    "positive_name": self._entity_name(positive_entity),
                    "positive_type": self._entity_type_by_id(positive_entity),
                    "negative_entity": negative_entity,
                    "negative_name": self._entity_name(negative_entity),
                    "negative_type": self._entity_type_by_id(negative_entity),
                    "policy_rank": int(rank_pos),
                    "prior_rank": int(prior_rank[col_idx].item()),
                    "sample_prob": float(sample_probs_cpu[row_idx, col_idx]) if sample_probs_cpu is not None else 0.0,
                    "base_score": float(row_components["base"][col_idx]),
                    "residual_score": float(row_components["residual"][col_idx]),
                    "memory_score": float(row_components["memory"][col_idx]),
                    "full_score": float(row_components["full"][col_idx]),
                    "disc_positive_score": current_pos,
                    "disc_negative_score": float(current_neg_scores[col_idx]),
                    "disc_margin": float(current_pos - current_neg_scores[col_idx]),
                    "margin_violation_raw": float(current_neg_scores[col_idx] - current_pos),
                    "reward": float(rewards_cpu[row_idx, col_idx]),
                    "current_hardest": int(col_idx == current_hardest),
                    "prior_hardest": int(col_idx == prior_hardest),
                    "Chemical_overlap": int(overlap["Chemical_overlap"]),
                    "Metabolite_overlap": int(overlap["Metabolite_overlap"]),
                    "Protein_overlap": int(overlap["Protein_overlap"]),
                    "Reaction_overlap": int(overlap["Reaction_overlap"]),
                })
        return rows

    def _write_trace_epoch(self, epoch_id, rows):
        path = os.path.join(self.trace_output_dir, "negative_trace_epoch_%04d.csv" % int(epoch_id))
        fieldnames = [
            "epoch", "batch_id", "trace_query_idx", "source", "corruption_side",
            "head_id", "tail_id", "relation_id", "head_name", "tail_name", "relation_name", "relation_family",
            "anchor_entity", "anchor_name", "anchor_type",
            "positive_entity", "positive_name", "positive_type",
            "negative_entity", "negative_name", "negative_type",
            "policy_rank", "prior_rank", "sample_prob",
            "base_score", "residual_score", "memory_score", "full_score",
            "disc_positive_score", "disc_negative_score", "disc_margin", "margin_violation_raw", "reward",
            "current_hardest", "prior_hardest",
            "Chemical_overlap", "Metabolite_overlap", "Protein_overlap", "Reaction_overlap",
        ]
        with open(path, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda x: (x["trace_query_idx"], x["policy_rank"], x["negative_entity"])):
                writer.writerow(row)

    def train(self, num_epoch=1, batch_per_epoch=None,epoch_id=0):

        print("traintraintraintraintraintraintraintraintraintrain")
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()
        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)
                ### TODO reasoning.reasoning.TorchDrug
                loss, metric, pred = model(batch, point=epoch_id)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
                    
                # Release GPU memory.
                # torch.cuda.empty_cache()
                
            if self.scheduler:
                self.scheduler.step()

    ### TODO reasoning.reasoning.TorchDrug


    def gan_train_astar(self, num_epoch=1, gen=None, batch_per_epoch=None, n_ent=0, n_rel=0):
        model = self.model
        model.split = "train"

        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
                gen = nn.parallel.DistributedDataParallel(gen, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
                gen = nn.parallel.DistributedDataParallel(gen, find_unused_parameters=True)
        model.train()

        def set_to_list(data, flag=1):
            tmp = []
            for it in data:
                tmp.append(it)
            m = torch.stack(tmp).t()
            if flag:
                return [m[0].tolist(), m[1].tolist(), m[2].tolist()]
            else:
                return m

        src, dst, rel = set_to_list(self.train_set, 0)
        best_perf = 0
        avg_reward = self.gan_avg_reward
        generator_for_update = gen.module if hasattr(gen, "module") else gen
        task_for_negative = model.module if hasattr(model, "module") else model
        generator_config = getattr(generator_for_update, "config", None)
        if isinstance(generator_config, dict):
            reward_ema_beta = float(generator_config.get("reward_ema_beta", 0.95))
        else:
            reward_ema_beta = float(getattr(generator_config, "reward_ema_beta", 0.95))
        requested_n_sample = int(self._cfg_get(generator_config, "requested_n_sample", 65536))
        generator_temperature = float(self._cfg_get(generator_config, "generator_temperature", 1.0))
        enable_policy_gradient = bool(self._cfg_get(generator_config, "enable_policy_gradient", True))
        n_train = len(src)

        for epoch in self.meter(num_epoch):
            self.gan_global_epoch += 1
            epoch_id = self.gan_global_epoch
            epoch_started_at = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            # p = model._strict_negative(src, rel, dst)
            # print(p)
            # src_cand, rel_cand, dst_cand = corrupter.corrupt(src, rel, dst, keep_truth=False)
            # print("===============")
            # print(src_cand.shape)
            # print(src_cand)
            bdata = TensorDataset(src, dst, rel)
            sampler = torch_data.DistributedSampler(bdata, self.world_size, self.rank)
            dataloader = data.DataLoader(bdata, self.batch_size, sampler=sampler, num_workers=self.num_worker)
            batch_per_epoch = batch_per_epoch or len(dataloader)
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
            epoch_d_loss = 0.0
            epoch_reward = 0.0
            epoch_reward_count = 0
            epoch_reward_nonzero = 0
            policy_stats = defaultdict(float)
            trace_rows = []
            trace_seen = set()
            trace_this_epoch = self.structured_logging_enabled and self.trace_query_count > 0 and ((epoch + 1) % self.trace_every_n_epoch == 0)
            
            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)
                # print("batch[0]")
                # print(batch[0])
                # Let the generator select negative samples.
                # print("===============")
                # print(batch[5].shape)
                # print(batch[5])
                # num_negative is the strict-negative candidate pool size.
                num_negative = int(task_for_negative.num_negative)
                neg_index = model._strict_negative(batch[0], batch[1], batch[2])
                current_batch_size = batch[0].size(0)
                split_point = current_batch_size // 2
                h_index = batch[0].unsqueeze(-1).repeat(1, num_negative + 1)
                t_index = batch[1].unsqueeze(-1).repeat(1, num_negative + 1)
                r_index = batch[2].unsqueeze(-1).repeat(1, num_negative + 1)

                t_index[:split_point, 1:] = neg_index[:split_point]
                h_index[split_point:, 1:] = neg_index[split_point:]
                gen_step = gen.gen_step(h_index[:,1:], r_index[:,1:], t_index[:,1:], n_sample=requested_n_sample, temperature=generator_temperature)


                neg_src_smpl, neg_dst_smpl = next(gen_step)
                n_adv_smp = neg_src_smpl.size(1)
                # print("myshape")
                # print(src_smpl.shape)
                ##############################

                # def sample_from_batch(data, num_samples):
                #     """
                    
                    
                #     """
                #     batchsize, n = data.size()
                    
                #     indices = torch.stack([torch.randperm(n)[:num_samples] for _ in range(batchsize)])
                    
                #     indices = indices.to(data.device)
                    
                #     sampled_data = torch.gather(data, 1, indices)
                    
                #     return sampled_data

                # src_smpl = sample_from_batch(src_smpl, 32)
                # dst_smpl = sample_from_batch(dst_smpl, 32)

                ##############################

                # rel_smpl = batch[5][:, :n_adv_smp]
                # Prepend the positive sample to the candidate batch.
                # Candidate batch contains one positive plus sampled negatives.
                src_smpl = torch.cat((batch[0].unsqueeze(1), neg_src_smpl), dim=1)
                dst_smpl = torch.cat((batch[1].unsqueeze(1), neg_dst_smpl), dim=1)
                rel_smpl = r_index[:, :n_adv_smp + 1]
                neg_rel_smpl = rel_smpl[:, 1:]


                ### TODO reasoning.reasoning.TorchDrug reasoning.forward
                loss, metric, pred = model(batch, is_gan=True, gen=gen, point=epoch_id, src_smpl=src_smpl, dst_smpl = dst_smpl, rel_smpl = rel_smpl)
                # Generator rewards must align with the ranking-loss hinge objective.
                # Ranking loss optimizes max(0, margin - s_pos + s_neg).
                # The generator reward should be each negative sample margin violation.
                ### relu(s_neg - s_pos + margin)。
                # This prioritizes negatives that truly violate the discriminator margin.
                task_for_reward = model.module if hasattr(model, "module") else model
                if "ranking" in task_for_reward.criterion:
                    positive_scores = pred[:, :1].detach()
                    negative_scores = pred[:, 1:].detach()
                    rewards = torch.relu(negative_scores - positive_scores + task_for_reward.margin)
                else:
                    rewards = pred[:, 1:].detach()

                if enable_policy_gradient and hasattr(generator_for_update, "update_memory"):
                    generator_for_update.update_memory(neg_src_smpl, neg_rel_smpl, neg_dst_smpl, rewards)

                batch_reward_mean = rewards.mean().detach()
                if avg_reward is None:
                    avg_reward = batch_reward_mean
                advantages = rewards - avg_reward

                sample_info = getattr(generator_for_update, "last_sample_info", None)
                sample_probs = sample_info.get("sample_probs") if sample_info else None

                if self.structured_logging_enabled:
                    self._accumulate_tensor_stat(policy_stats, "reward", rewards)
                    hardest_reward = rewards.max(dim=-1).values
                    self._accumulate_tensor_stat(policy_stats, "hardest_reward", hardest_reward)
                    self._accumulate_tensor_stat(policy_stats, "hardest_reward_nonzero", (hardest_reward > 0).float())

                    if sample_info:
                        self._accumulate_tensor_stat(policy_stats, "policy_entropy", sample_info.get("entropy"))
                        self._accumulate_tensor_stat(policy_stats, "sample_prob", sample_info.get("sample_probs"))
                        self._accumulate_scalar_mean(policy_stats, "effective_sample_size", sample_info.get("sample_size", 0))

                    if self.policy_metric_sample_size > 0 and hasattr(generator_for_update, "score_components"):
                        probe_size = min(self.policy_metric_sample_size, n_adv_smp)
                        if probe_size > 0:
                            with torch.no_grad():
                                probe_components = generator_for_update.score_components(
                                    neg_src_smpl[:, :probe_size],
                                    neg_rel_smpl[:, :probe_size],
                                    neg_dst_smpl[:, :probe_size]
                                )
                            self._accumulate_scalar_mean(policy_stats, "component_probe_size", probe_size)
                            self._accumulate_tensor_stat(policy_stats, "base_score", probe_components["base"])
                            self._accumulate_tensor_stat(policy_stats, "residual_score", probe_components["residual"])
                            self._accumulate_tensor_stat(policy_stats, "memory_score", probe_components["memory"])
                            self._accumulate_tensor_stat(policy_stats, "full_score", probe_components["full"])
                            self._accumulate_tensor_stat(policy_stats, "memory_active", (probe_components["memory"].abs() > 1e-6).float())
                            self._accumulate_tensor_stat(policy_stats, "adapter_scale_effective", probe_components["adapter_scale_effective"])
                            self._accumulate_tensor_stat(policy_stats, "memory_scale_effective", probe_components["memory_scale_effective"])

                    row_index = torch.arange(neg_src_smpl.size(0), device=neg_src_smpl.device)
                    corrupted_entity = neg_dst_smpl.clone()
                    head_corrupt_mask = row_index >= split_point
                    corrupted_entity[head_corrupt_mask] = neg_src_smpl[head_corrupt_mask]
                    unique_ratio = torch.unique(corrupted_entity.reshape(-1)).numel() / max(corrupted_entity.numel(), 1)
                    self._accumulate_scalar_mean(policy_stats, "sampled_entity_unique_ratio", unique_ratio)

                    if trace_this_epoch:
                        trace_rows.extend(
                            self._collect_trace_rows(
                                epoch + 1, batch_id, batch, neg_src_smpl, neg_dst_smpl, neg_rel_smpl,
                                pred, rewards, sample_probs, generator_for_update, trace_seen
                            )
                        )

                # Keep rewards as [batch, sample]; do not unsqueeze them.
                # Otherwise reward and log_prob broadcast to [batch, batch, sample].
                # That would mix rewards across queries and corrupt REINFORCE gradients.
                if enable_policy_gradient:
                    gen_step.send(advantages)
                    avg_reward = reward_ema_beta * avg_reward + (1 - reward_ema_beta) * batch_reward_mean
                else:
                    gen_step.send(None)

                epoch_reward += rewards.sum().item()
                epoch_reward_count += rewards.numel()
                epoch_reward_nonzero += (rewards > 0).sum().item()
                sample_info = getattr(generator_for_update, "last_sample_info", None) or {}
                update_info = getattr(generator_for_update, "last_update_info", None) or {}
                self._accumulate_scalar_mean(policy_stats, "policy_update_enabled", float(enable_policy_gradient))
                self._accumulate_scalar_mean(policy_stats, "generator_grad_norm", update_info.get("generator_grad_norm", 0.0))
                self._accumulate_scalar_mean(policy_stats, "reinforce_loss", update_info.get("reinforce_loss", 0.0))
                self._accumulate_scalar_mean(policy_stats, "invalid_prob_row_count", sample_info.get("invalid_prob_rows", 0))
                self._accumulate_scalar_mean(policy_stats, "invalid_prob_row_ratio", sample_info.get("invalid_prob_row_ratio", 0.0))
                self._accumulate_scalar_mean(policy_stats, "logit_min", sample_info.get("logit_min", 0.0))
                self._accumulate_scalar_mean(policy_stats, "logit_max", sample_info.get("logit_max", 0.0))
                self._accumulate_scalar_mean(policy_stats, "logit_mean", sample_info.get("logit_mean", 0.0))
                self._accumulate_scalar_mean(policy_stats, "logit_std", sample_info.get("logit_std", 0.0))
                generator_for_update.last_sample_info = None
                generator_for_update.last_update_info = None
                
                
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                # Track discriminator loss after the loss computation.
                epoch_d_loss += loss.item()
                loss.backward()
                discriminator_grad_norm = self._model_grad_norm(self.model.parameters())
                self._accumulate_scalar_mean(policy_stats, "discriminator_grad_norm", discriminator_grad_norm)
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
            if self.scheduler:
                self.scheduler.step()
            avg_loss = epoch_d_loss / max(batch_per_epoch, 1)
            avg_reward_log = epoch_reward / max(epoch_reward_count, 1)
            reward_nonzero_ratio = epoch_reward_nonzero / max(epoch_reward_count, 1)
            baseline_value = avg_reward.item() if avg_reward is not None else 0.0
            self.gan_avg_reward = avg_reward.detach() if avg_reward is not None else None
            epoch_seconds = time.time() - epoch_started_at
            gpu_peak_mem_mb = 0.0
            if self.device.type == "cuda":
                gpu_peak_mem_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
            logging.info(
                'Epoch %d/%d, D_loss=%f, reward=%f, reward_nonzero=%f, ema_baseline=%f',
                epoch + 1, num_epoch, avg_loss, avg_reward_log, reward_nonzero_ratio, baseline_value
            )
            if self.structured_logging_enabled:
                train_fieldnames = [
                    "epoch", "discriminator_loss", "reward_mean", "reward_nonzero_ratio", "ema_baseline",
                    "batch_size", "num_negative", "gan_sample_cap", "requested_n_sample", "effective_sample_size",
                    "generator_temperature", "gpu_peak_mem_mb", "epoch_seconds",
                    "policy_update_enabled", "generator_grad_norm", "discriminator_grad_norm", "reinforce_loss",
                    "invalid_prob_row_count", "invalid_prob_row_ratio",
                    "logit_min", "logit_max", "logit_mean", "logit_std"
                ]
                generator_config = getattr(generator_for_update, "config", None)
                gan_sample_cap = int(self._cfg_get(generator_config, "gan_sample_cap", 131072))
                effective_sample_size = policy_stats.get("effective_sample_size_sum", 0.0) / max(policy_stats.get("effective_sample_size_count", 0.0), 1.0)
                policy_update_enabled_mean = policy_stats.get("policy_update_enabled_sum", 0.0) / max(policy_stats.get("policy_update_enabled_count", 0.0), 1.0)
                generator_grad_norm_mean = policy_stats.get("generator_grad_norm_sum", 0.0) / max(policy_stats.get("generator_grad_norm_count", 0.0), 1.0)
                discriminator_grad_norm_mean = policy_stats.get("discriminator_grad_norm_sum", 0.0) / max(policy_stats.get("discriminator_grad_norm_count", 0.0), 1.0)
                reinforce_loss_mean = policy_stats.get("reinforce_loss_sum", 0.0) / max(policy_stats.get("reinforce_loss_count", 0.0), 1.0)
                invalid_prob_row_count_mean = policy_stats.get("invalid_prob_row_count_sum", 0.0) / max(policy_stats.get("invalid_prob_row_count_count", 0.0), 1.0)
                invalid_prob_row_ratio_mean = policy_stats.get("invalid_prob_row_ratio_sum", 0.0) / max(policy_stats.get("invalid_prob_row_ratio_count", 0.0), 1.0)
                logit_min_mean = policy_stats.get("logit_min_sum", 0.0) / max(policy_stats.get("logit_min_count", 0.0), 1.0)
                logit_max_mean = policy_stats.get("logit_max_sum", 0.0) / max(policy_stats.get("logit_max_count", 0.0), 1.0)
                logit_mean_mean = policy_stats.get("logit_mean_sum", 0.0) / max(policy_stats.get("logit_mean_count", 0.0), 1.0)
                logit_std_mean = policy_stats.get("logit_std_sum", 0.0) / max(policy_stats.get("logit_std_count", 0.0), 1.0)
                self._append_csv_row(self.train_epoch_metrics_file, train_fieldnames, {
                    "epoch": int(epoch + 1),
                    "discriminator_loss": float(avg_loss),
                    "reward_mean": float(avg_reward_log),
                    "reward_nonzero_ratio": float(reward_nonzero_ratio),
                    "ema_baseline": float(baseline_value),
                    "batch_size": int(self.batch_size),
                    "num_negative": int(task_for_negative.num_negative),
                    "gan_sample_cap": gan_sample_cap,
                    "requested_n_sample": int(requested_n_sample),
                    "effective_sample_size": float(effective_sample_size),
                    "generator_temperature": float(generator_temperature),
                    "gpu_peak_mem_mb": float(gpu_peak_mem_mb),
                    "epoch_seconds": float(epoch_seconds),
                    "policy_update_enabled": float(policy_update_enabled_mean),
                    "generator_grad_norm": float(generator_grad_norm_mean),
                    "discriminator_grad_norm": float(discriminator_grad_norm_mean),
                    "reinforce_loss": float(reinforce_loss_mean),
                    "invalid_prob_row_count": float(invalid_prob_row_count_mean),
                    "invalid_prob_row_ratio": float(invalid_prob_row_ratio_mean),
                    "logit_min": float(logit_min_mean),
                    "logit_max": float(logit_max_mean),
                    "logit_mean": float(logit_mean_mean),
                    "logit_std": float(logit_std_mean),
                })

                base_mean, base_std = self._finalize_stat(policy_stats, "base_score")
                residual_mean, residual_std = self._finalize_stat(policy_stats, "residual_score")
                memory_mean, memory_std = self._finalize_stat(policy_stats, "memory_score")
                full_mean, full_std = self._finalize_stat(policy_stats, "full_score")
                entropy_mean, entropy_std = self._finalize_stat(policy_stats, "policy_entropy")
                sample_prob_mean, sample_prob_std = self._finalize_stat(policy_stats, "sample_prob")
                reward_mean, reward_std = self._finalize_stat(policy_stats, "reward")
                hardest_mean, hardest_std = self._finalize_stat(policy_stats, "hardest_reward")
                hardest_nonzero_mean, _ = self._finalize_stat(policy_stats, "hardest_reward_nonzero")
                memory_active_mean, _ = self._finalize_stat(policy_stats, "memory_active")
                component_probe_size = policy_stats.get("component_probe_size_sum", 0.0) / max(policy_stats.get("component_probe_size_count", 0.0), 1.0)
                sampled_entity_unique_ratio = policy_stats.get("sampled_entity_unique_ratio_sum", 0.0) / max(policy_stats.get("sampled_entity_unique_ratio_count", 0.0), 1.0)
                policy_fieldnames = [
                    "epoch", "component_probe_size", "base_score_mean", "base_score_std",
                    "residual_score_mean", "residual_score_std", "memory_score_mean", "memory_score_std",
                    "full_score_mean", "full_score_std", "policy_entropy_mean", "policy_entropy_std",
                    "sample_prob_mean", "sample_prob_std", "reward_mean", "reward_std",
                    "hardest_reward_mean", "hardest_reward_std", "hardest_reward_nonzero_ratio",
                    "sampled_entity_unique_ratio", "memory_active_ratio", "adapter_scale", "memory_scale",
                    "head_memory_norm", "tail_memory_norm", "adapter_scale_effective", "memory_scale_effective"
                ]
                head_memory_norm = float(generator_for_update.mdl.head_memory_bank.norm().item()) if hasattr(generator_for_update.mdl, "head_memory_bank") else 0.0
                tail_memory_norm = float(generator_for_update.mdl.tail_memory_bank.norm().item()) if hasattr(generator_for_update.mdl, "tail_memory_bank") else 0.0
                adapter_scale = float(generator_for_update.mdl.adapter_scale.detach().item()) if hasattr(generator_for_update.mdl, "adapter_scale") else 0.0
                memory_scale = float(generator_for_update.mdl.memory_scale.detach().item()) if hasattr(generator_for_update.mdl, "memory_scale") else 0.0
                adapter_scale_eff, _ = self._finalize_stat(policy_stats, "adapter_scale_effective")
                memory_scale_eff, _ = self._finalize_stat(policy_stats, "memory_scale_effective")
                self._append_csv_row(self.policy_epoch_metrics_file, policy_fieldnames, {
                    "epoch": int(epoch + 1),
                    "component_probe_size": float(component_probe_size),
                    "base_score_mean": float(base_mean),
                    "base_score_std": float(base_std),
                    "residual_score_mean": float(residual_mean),
                    "residual_score_std": float(residual_std),
                    "memory_score_mean": float(memory_mean),
                    "memory_score_std": float(memory_std),
                    "full_score_mean": float(full_mean),
                    "full_score_std": float(full_std),
                    "policy_entropy_mean": float(entropy_mean),
                    "policy_entropy_std": float(entropy_std),
                    "sample_prob_mean": float(sample_prob_mean),
                    "sample_prob_std": float(sample_prob_std),
                    "reward_mean": float(reward_mean),
                    "reward_std": float(reward_std),
                    "hardest_reward_mean": float(hardest_mean),
                    "hardest_reward_std": float(hardest_std),
                    "hardest_reward_nonzero_ratio": float(hardest_nonzero_mean),
                    "sampled_entity_unique_ratio": float(sampled_entity_unique_ratio),
                    "memory_active_ratio": float(memory_active_mean),
                    "adapter_scale": float(adapter_scale),
                    "memory_scale": float(memory_scale),
                    "head_memory_norm": float(head_memory_norm),
                    "tail_memory_norm": float(tail_memory_norm),
                    "adapter_scale_effective": float(adapter_scale_eff),
                    "memory_scale_effective": float(memory_scale_eff),
                })
                if trace_this_epoch:
                    self._write_trace_epoch(epoch + 1, trace_rows)


    # def gan_train_astar(self, num_epoch=1, gen=None, batch_per_epoch=None, n_ent=0, n_rel=0):
    #     model = self.model
    #     model.split = "train"
    #
    #     if self.world_size > 1:
    #         if self.device.type == "cuda":
    #             model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
    #                                                         find_unused_parameters=True)
    #             gen = nn.parallel.DistributedDataParallel(gen, device_ids=[self.device],
    #                                                         find_unused_parameters=True)
    #         else:
    #             model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    #             gen = nn.parallel.DistributedDataParallel(gen, find_unused_parameters=True)
    #     model.train()
    #
    #     def set_to_list(data, flag=1):
    #         tmp = []
    #         print(tmp)
    #         for it in data:
    #             tmp.append(it)
    #         m = torch.stack(tmp).t()
    #         if flag:
    #             return [m[0].tolist(), m[1].tolist(), m[2].tolist()]
    #         else:
    #             return m
    #
    #     # Build the candidate sample set.
    #     corrupter = BernCorrupterMulti(set_to_list(self.train_set), n_ent, n_rel, 2000)
    #     src, dst, rel = set_to_list(self.train_set, 0)
    #     best_perf = 0
    #     avg_reward = 0
    #     n_train = len(src)
    #     for epoch in self.meter(num_epoch):
    #         # p = model._strict_negative(src, rel, dst)
    #         # print(p)
    #         src_cand, rel_cand, dst_cand = corrupter.corrupt(src, rel, dst, keep_truth=False)
    #         # print("===============")
    #         # print(src_cand.shape)
    #         # print(src_cand)
    #         bdata = TensorDataset(src, dst, rel, src_cand, dst_cand, rel_cand)
    #         sampler = torch_data.DistributedSampler(bdata, self.world_size, self.rank)
    #         dataloader = data.DataLoader(bdata, self.batch_size, sampler=sampler, num_workers=self.num_worker)
    #         batch_per_epoch = batch_per_epoch or len(dataloader)
    #         sampler.set_epoch(epoch)
    #
    #         metrics = []
    #         start_id = 0
    #         # the last gradient update may contain less than gradient_interval batches
    #         gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
    #         epoch_d_loss = 0
    #         epoch_reward = 0
    #         n_adv_smp = 32
    #         point = 1
    #         for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
    #             if self.device.type == "cuda":
    #                 batch = utils.cuda(batch, device=self.device)
    #             # print("batch[0]")
    #             # print(batch[0])
    #             # Let the generator select negatives.
    #             # print("===============")
    #             # print(batch[5].shape)
    #             # print(batch[5])
    #             gen_step = gen.gen_step(batch[3], batch[5], batch[4], n_sample=n_adv_smp, temperature=0.5)
    #             src_smpl, dst_smpl = next(gen_step)
    #             rel_smpl = batch[5][:, :n_adv_smp]
    #             # Prepend the positive sample.
    #             # Candidate batch contains one positive plus sampled negatives.
    #             src_smpl = torch.cat((batch[0].unsqueeze(1), src_smpl), dim=1)
    #             dst_smpl = torch.cat((batch[1].unsqueeze(1), dst_smpl), dim=1)
    #             rel_smpl = torch.cat((batch[2].unsqueeze(1), rel_smpl), dim=1)
    #
    #             batch[3] = src_smpl
    #             batch[4] = dst_smpl
    #             batch[5] = rel_smpl
    #
    #             ### TODO reasoning.reasoning.TorchDrug reasoning.forward
    #             loss, metric, pred = model(batch, is_gan=True, gen=gen, point=point)
    #             point = point + 1
    #             # Update generator gradients.
    #             rewards = pred[:, 1:]
    #             # Accumulate epoch rewards.
    #             epoch_reward += torch.sum(rewards)
    #             rewards = rewards - avg_reward
    #             gen_step.send(rewards.unsqueeze(1))
    #             if not loss.requires_grad:
    #                 raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
    #             loss = loss / gradient_interval
    #             # Track discriminator loss after loss computation.
    #             epoch_d_loss += torch.sum(loss)
    #             loss.backward()
    #             metrics.append(metric)
    #
    #             if batch_id - start_id + 1 == gradient_interval:
    #                 self.optimizer.step()
    #                 self.optimizer.zero_grad()
    #
    #                 metric = utils.stack(metrics, dim=0)
    #                 metric = utils.mean(metric, dim=0)
    #                 if self.world_size > 1:
    #                     metric = comm.reduce(metric, op="mean")
    #                 self.meter.update(metric)
    #
    #                 metrics = []
    #                 start_id = batch_id + 1
    #                 gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
    #
    #         if self.scheduler:
    #             self.scheduler.step()
    #         avg_loss = epoch_d_loss / n_train
    #         avg_reward = epoch_reward / n_train
    #         logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, num_epoch, avg_loss, avg_reward)

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        ### self.batch_size//2
        dataloader = data.DataLoader(test_set, 48, sampler=sampler, num_workers=self.num_worker)
        model = self.model
        model.split = split
        model.eval()
        # preds = []
        # targets = []
        metric_sum = {}
        metric_weight = 0.0
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)
            pred, target = model.predict_and_target(batch)
            metric = model.evaluate(pred, target)
            if isinstance(target, tuple):
                weight = float(target[1].numel())
            else:
                weight = float(target.numel())
            for key, value in metric.items():
                metric_sum[key] = metric_sum.get(key, 0.0) + float(value.detach().item()) * weight
            metric_weight += weight
            # Release batch-level tensors during evaluation to avoid validation-time memory pressure.
            del batch, pred, target, metric
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Compute final metrics.
        if metric_weight > 0:
            final_metrics = {key: value / metric_weight for key, value in metric_sum.items()}
        else:
            final_metrics = {}

        if log:
            self.meter.log(final_metrics, category="%s/epoch" % split)

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return final_metrics

    def get_gan_resume_state(self):
        if self.gan_avg_reward is None:
            avg_reward = None
        else:
            avg_reward = self.gan_avg_reward.detach().cpu()
        return {
            "gan_avg_reward": avg_reward,
            "gan_global_epoch": int(self.gan_global_epoch),
        }

    def load_gan_resume_state(self, state):
        avg_reward = state.get("gan_avg_reward")
        if avg_reward is None:
            self.gan_avg_reward = None
        else:
            self.gan_avg_reward = avg_reward.to(self.device)
        self.gan_global_epoch = int(state.get("gan_global_epoch", 0))

    def load(self, checkpoint, load_optimizer=True, strict=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"], strict=strict)
        # self.model.load_state_dict(state["model"], strict=False)
        

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()

    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError("Expect config class to be `%s`, but found `%s`" % (cls.__name__, config["class"]))

        optimizer_config = config.pop("optimizer")
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        optimizer_config["params"] = new_config["task"].parameters()
        new_config["optimizer"] = core.Configurable.load_config_dict(optimizer_config)

        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
