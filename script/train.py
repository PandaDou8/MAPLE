import logging
import os
import gc
# os.environ['TORCH_CUDA_ARCH_LIST'] = ''
import sys
import math
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import torch
import random
import numpy as np

# from reasoning.TorchDrug import core, tasks

# sys.path.append：这行代码将脚本所在目录的父目录的父目录添加到系统路径中，以便能够导入reasoning模块。
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util
from reasoning.TorchDrug import core
# from reasoning import task1
from reasoning.TorchDrug.utils import comm, pretty
import pretrain
from config import overwrite_config_with_args, dump_config, config
from memory_distmult import MemoryAugmentedDistMult


def _set_single_gpu_from_cfg(cfg):
    # 0424 FIX: 单卡训练统一从 YAML 的 engine.gpus[0] 读取，不再使用脚本里的硬编码 GPU
    if not torch.cuda.is_available():
        return
    if "engine" in cfg and "gpus" in cfg.engine and cfg.engine.gpus:
        torch.cuda.set_device(int(cfg.engine.gpus[0]))


def _build_generator(dataset):
    gen_name = config().g_config
    gen_config = config()[gen_name]

    if gen_name == "DistMult":
        return MemoryAugmentedDistMult(dataset.num_entity, dataset.num_relation, gen_config)

    raise ValueError("Unsupported generator config `%s`; MAPLE_main only keeps the DistMult/MAPLE generator." % gen_name)


def _resume_epoch_numbering_if_needed(cfg, solver):
    # 0427 RESUME FIX: 续训时只恢复 epoch 编号，不改旧目录中的文件。
    # 这样新的时间戳目录会从 resume_epoch + 1 开始保存 checkpoint，例如 38、39...
    resume_epoch = int(cfg.get("resume_epoch", 0) if isinstance(cfg, dict) else getattr(cfg, "resume_epoch", 0) or 0)
    if resume_epoch > 0:
        solver.meter.epoch_id = resume_epoch
        # 0427 RESUME FIX 2: Meter 不只依赖 epoch_id，还依赖 epoch2batch / time 的历史长度。
        # 如果只改 epoch_id，epoch 结束时计算 ETA 会访问 self.time[self.start_epoch] 并越界。
        # 这里用当前启动时刻补齐占位历史，既能继续编号，也不会污染本次 epoch 的均值统计切片。
        meter = solver.meter
        history_len = resume_epoch + 1
        current_time = meter.time[-1]
        meter.epoch2batch = [0] * history_len
        meter.time = [current_time] * history_len
        logging.info("Resume checkpoint numbering from epoch %d", resume_epoch)


def _cleanup_runtime_memory():
    # 0427 OOM FIX: 只做安全的显存/对象回收，不改变训练结果语义。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generator_resume_path(epoch):
    return "generator_resume_epoch_%d.pth" % epoch


def _training_resume_path(epoch):
    return "training_resume_epoch_%d.pth" % epoch


def _save_training_resume_state(path, solver, epoch, best_epoch, best_result):
    state = {
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_result": float(best_result),
        "engine_gan_state": solver.get_gan_resume_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, path)


def _load_training_resume_state(path, solver):
    state = torch.load(path, map_location="cpu")
    engine_state = state.get("engine_gan_state", {})
    solver.load_gan_resume_state(engine_state)
    python_state = state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return state


# train_and_validate函数用于训练和验证模型。
# 它接受两个参数：cfg和solver。
def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    if hasattr(cfg.train, "batch_per_epoch"):
        step = 3
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1
    # solver.load(config().pretrain_astar)
    # solver.evaluate("valid")
    for i in range(0, cfg.train.num_epoch, 1):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(1, cfg.train.num_epoch - i)

        solver.train(**kwargs, epoch_id=i)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver

### pre_train_and_validate函数用于预训练和验证模型。
#### 还没改
def gan_train_and_validate(cfg, solver,dataset):

    gen = _build_generator(dataset)

    ### 加载预训练生成器
    # gen_model_path = config().pretrain_dir + config().dataset["class"] + "/" + config().pretrain_config + ".mdl"
    gen_model_path = config().pretrain_gen_model
    generator_resume_state = getattr(cfg, "generator_resume_state", None)
    resume_state_path = getattr(cfg, "resume_state", None)

    if generator_resume_state:
        logging.info("读取生成器完整续训状态，路径为：" + generator_resume_state)
        gen.load_training_state(generator_resume_state)
    elif(os.path.exists(gen_model_path)==False):
        logging.warn("生成器预训练模型不存在，路径为："+gen_model_path)
        logging.info("开始预训练生成器......")
        pretrain.gen_pretrain()
    else:
        logging.info("读取生成器初始化模型，路径为："+gen_model_path)
        gen.load(gen_model_path)

    if resume_state_path:
        logging.info("读取训练续训状态，路径为：" + resume_state_path)
        resume_state = _load_training_resume_state(resume_state_path, solver)
        best_result = float(resume_state.get("best_result", float("-inf")))
        best_epoch = int(resume_state.get("best_epoch", -1))
    else:
        best_result = float("-inf")
        best_epoch = -1
    
    # print("#####################test gen#####################")
    # pretrain.test_gen(gen)

    if cfg.train.num_epoch == 0:
        return

    ### 设置测试epoch数
    if hasattr(cfg.train, "batch_per_epoch"):
        step = 3
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    # logging.info("读取判别器："+config().pretrain_astar)
    # solver.load(config().pretrain_astar)
    
    # print("#####################valid dis#####################")
    # solver.evaluate("valid")
    
    # print("#####################test dis#####################")
    # solver.evaluate("test")
    
    # print(1/0)
    
    
    # solver.evaluate("valid")
    ### 训练num_epoch次，步长为step
    for i in range(0, cfg.train.num_epoch, 1):
        kwargs = cfg.train.copy()
        ### 以step为模型训练周期
        kwargs["num_epoch"] = min(1, cfg.train.num_epoch - i)
        solver.gan_train_astar(**kwargs,gen=gen,n_ent=dataset.num_entity,n_rel=dataset.num_relation)
        solver.save("model_pretrain_epoch_%d.pth" % solver.epoch)
        # Only keep full generator resume checkpoints. Plain generator-only checkpoints
        # duplicated the exact same model weights without optimizer / RNG / EMA state.
        gen.save_training_state(_generator_resume_path(solver.epoch), extra={"epoch": int(solver.epoch)})
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        del metric
        _cleanup_runtime_memory()
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
            gen.save_training_state("generator_resume_best.pth", extra={"epoch": int(solver.epoch), "best_epoch": int(best_epoch)})
        _save_training_resume_state(
            _training_resume_path(solver.epoch),
            solver,
            epoch=solver.epoch,
            best_epoch=best_epoch,
            best_result=best_result,
        )
        if solver.epoch == best_epoch:
            _save_training_resume_state(
                "training_resume_best.pth",
                solver,
                epoch=solver.epoch,
                best_epoch=best_epoch,
                best_result=best_result,
            )

    solver.load("model_pretrain_epoch_%d.pth" % best_epoch)
    return solver


# test函数用于测试模型。
def test(cfg, solver):
    solver.evaluate("valid")
    solver.evaluate("test")

if __name__ == "__main__":
    args, vars = util.parse_args() # 使用util.parse_args解析命令行参数
    cfg = util.load_config(args.config, context=vars)
    _set_single_gpu_from_cfg(cfg)
    working_dir = util.create_working_directory(cfg)

    # torch.manual_seed(args.seed + comm.get_rank())
    
    seed = args.seed + comm.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pretty.format(cfg))


    dataset = core.Configurable.load_config_dict(cfg.dataset)

    print("==================")
    # print(dataset.triplets)
    #### 这个模型没有加双向边
    # WN18RR(
    #     # entity: 40943
    #     # relation: 11
    #     # triplet: 93003
    # )


    solver = util.build_solver(cfg, dataset)
    _resume_epoch_numbering_if_needed(cfg, solver)

    if cfg.train.num_epoch > 0:
        logging.info("对抗训练......")
        gan_train_and_validate(cfg, solver, dataset)
    else:
        logging.info("直接评估模型......")
        solver.load(config().pretrain_astar)

    test(cfg, solver)
