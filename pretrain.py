import os
import sys
import torch

from reasoning.TorchDrug import core
from reasoning.TorchDrug.utils import comm, pretty

from config import overwrite_config_with_args, dump_config, config
from corrupter import BernCorrupter, BernCorrupterMulti
from data_utils import heads_tails, set_to_list
from distmult import DistMult
from logger_init import logger_init

# sys.path.append：这行代码将脚本所在目录的父目录的父目录添加到系统路径中，以便能够导入reasoning模块。
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util
###### 预训练生成器 ######


def _set_single_gpu_from_cfg(cfg):
    # 0424 FIX: 单卡训练统一从 YAML 的 engine.gpus[0] 读取，不再使用脚本里的硬编码 GPU
    if not torch.cuda.is_available():
        return
    if "engine" in cfg and "gpus" in cfg.engine and cfg.engine.gpus:
        torch.cuda.set_device(int(cfg.engine.gpus[0]))

_PRETRAIN_CONTEXT = None


def _build_pretrain_context():
    args, vars = util.parse_args()  # 使用util.parse_args解析命令行参数
    cfg = util.load_config(args.config, context=vars)
    _set_single_gpu_from_cfg(cfg)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pretty.format(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_data, valid_data, test_data = dataset.split()
    n_ent = dataset.num_entity
    n_rel = dataset.num_relation
    train_data = set_to_list(train_data)
    valid_data = set_to_list(valid_data)
    test_data = set_to_list(test_data)

    heads, tails = heads_tails(n_ent, train_data, valid_data, test_data)

    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data = [torch.LongTensor(vec) for vec in test_data]
    train_data = [torch.LongTensor(vec) for vec in train_data]

    mdl_type = config().pretrain_config
    gen_config = config()[mdl_type]
    if mdl_type == 'DistMult':
        corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, gen_config.n_sample)
        gen = DistMult(n_ent, n_rel, gen_config)
    else:
        raise ValueError("Unsupported pretrain config `%s`; MAPLE_main only keeps DistMult pretraining." % mdl_type)

    tester = lambda: gen.test_link(valid_data, n_ent, heads, tails)
    return {
        "args": args,
        "cfg": cfg,
        "working_dir": working_dir,
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
        "n_ent": n_ent,
        "heads": heads,
        "tails": tails,
        "tester": tester,
        "corrupter": corrupter,
        "gen": gen,
    }


def _get_pretrain_context():
    global _PRETRAIN_CONTEXT
    if _PRETRAIN_CONTEXT is None:
        _PRETRAIN_CONTEXT = _build_pretrain_context()
    return _PRETRAIN_CONTEXT

# gen.pretrain(train_data, corrupter, tester)
# gen.load(os.path.join(task_dir, gen_config.model_file))
# gen.load(os.path.join("/home/pdou/experiments/modeltest/0_path_model/AStarNet-master/experiments/WN18RR/DistMult.mdl"))
# gen.test_link(test_data, n_ent, heads, tails)


def gen_pretrain():
    ctx = _get_pretrain_context()
    gen = ctx["gen"]
    gen.pretrain(ctx["train_data"], ctx["corrupter"], ctx["tester"])
    gen.test_link(ctx["test_data"], ctx["n_ent"], ctx["heads"], ctx["tails"])

def test_gen(gen):
    ctx = _get_pretrain_context()
    gen.test_link(ctx["test_data"], ctx["n_ent"], ctx["heads"], ctx["tails"])

if __name__ == '__main__':
    gen_pretrain()
