import os
import time
import logging
import argparse
import uuid

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from reasoning.TorchDrug import core, utils
from reasoning.TorchDrug.utils import comm

import pprint


logger = logging.getLogger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context or {})
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def create_working_directory(cfg):
    sync_file = os.path.join(os.path.expanduser(cfg.output_dir), ".working_dir_%s.tmp" % uuid.uuid4().hex)
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    base_working_dir = os.path.join(
        os.path.expanduser(cfg.output_dir),
        cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
        time.strftime("%Y-%m-%d-%H-%M-%S")
    )
    working_dir = base_working_dir

    # synchronize working directory
    if comm.get_rank() == 0:
        os.makedirs(os.path.expanduser(cfg.output_dir), exist_ok=True)
        if os.path.exists(working_dir):
            working_dir = "%s-%s" % (base_working_dir, uuid.uuid4().hex[:8])
        with open(sync_file, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir, exist_ok=False)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(sync_file, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(sync_file)

    os.chdir(working_dir)
    return working_dir


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def build_solver(cfg, dataset):

    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    ### 缩小测试集和验证集
    if "fast_test" in cfg and cfg.fast_test is not None:
        if comm.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid / test." % cfg.fast_test)
        g = torch.Generator()
        g.manual_seed(1024)
        valid_set = torch_data.random_split(valid_set, [cfg.fast_test, len(valid_set) - cfg.fast_test], generator=g)[0]
        test_set = torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]
    cfg.task.model.base_layer.num_relation = int(dataset.num_relation)
    ### 读取配置和优化器
    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    ### 构建模型
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    ### 加载checkpoint
    if "checkpoint" in cfg:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.checkpoint)
        checkpoint = os.path.expanduser(cfg.checkpoint)
        state = torch.load(cfg.checkpoint, map_location=solver.device)
        state["model"] = {k: v for k, v in state["model"].items() if isinstance(v, torch.Tensor)}

        solver.model.load_state_dict(state["model"], strict=False)
        solver.optimizer.load_state_dict(state["optimizer"])
        for state in solver.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(solver.device)

        comm.synchronize()

    # TODO 分布式训练的逻辑还没有搞清楚，comm是获得第一个设备上的吗？判断是不是第一个也是为了不重复操作是这样吗?
    # 打印模型结构
    if comm.get_rank() == 0:
        logger.warning("Model Structure_bulit solver:")
        pprint.pprint(solver.model)

    return solver


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars
