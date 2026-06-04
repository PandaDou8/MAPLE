import os
import sys
import pprint

import torch
torch.cuda.set_device(1)


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util
from reasoning.TorchDrug import core
from reasoning.TorchDrug.utils import comm

def load_vocab(dataset): # 加载词汇表
    path = '/dataStor/home/yqzhang/data/mic_data_0523/'
    vocabs = []
    for object in ["entity", "relation"]: # 加载实体和关系的词汇表
        vocab_file = os.path.join(path,"%s.txt" % object)
        mapping = {}
        with open(vocab_file, "r") as fin:
            for line in fin:
                k, v = line.strip().split("\t")
                mapping[v] = k
        # vocab = [mapping[t] for t in getattr(dataset, "%s_vocab" % object)]
        vocab = list(mapping.values())
        vocabs.append(vocab)

    return vocabs # 返回实体和关系的词汇表



def rank(solver, sample, entity_vocab, relation_vocab, t_entities):
    num_relation = len(relation_vocab)
    h_index, t_index, r_index = sample.unbind(-1) # 分离头实体、尾实体和关系
    inverse = torch.stack([t_index, h_index, r_index + num_relation], dim=-1) # 生成反向三元组
    batch = sample.unsqueeze(0) # 将样本扩展为批次
    if sample.ndim == 1: # 如果样本是一维的，则将其扩展为批次
        vis_batch = torch.stack([sample])
        # vis_batch = torch.stack([sample, inverse]) # 生成正向和反向三元组的批次
    else:
        is_t_neg = (h_index == h_index[0]).all() # 判断是否为尾实体负例
        vis_batch = sample[:1] if is_t_neg else inverse[:1] # 如果是尾实体负例，则选择正向三元组，否则选择反向三元组
    batch = batch.to(solver.device) 
    vis_batch = vis_batch.to(solver.device)
    solver.model.eval()
    with torch.no_grad():
        # pred, target = solver.model.predict_and_target(batch,t_entities) # 预测和目标
        pred, target = solver.model.vis_predict_and_target(batch,t_entities) # 预测和目标
    
    #### 这个map是用来实现通过全局index找到在类别中的位置
    t_entities_map = {word: idx for idx, word in enumerate(t_entities)}
    
    if isinstance(target, tuple):
        mask, target = target # 如果目标是一个元组，则分离掩码和目标
        tmp = t_entities_map[target.squeeze().item()]
        tmp = torch.tensor([[[tmp]]])
      
        #  需要建立一个target到结果下标的映射
        # pos_pred = pred.gather(-1, target.unsqueeze(-1)) # 获取正例预测值
        pos_pred = pred.gather(-1, tmp) # 获取正例预测值
        pred_squeezed = pred.squeeze()  # 形状从 [1,1,31332] -> [31332]
        # 2. 组合成二元组（实体索引，预测值）
        pairs = list(zip(t_entities, pred_squeezed.tolist()))
        
        
        # 3. 按 pred 的值升序排列
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        # print(pairs_sorted)
        # 4. 输出前500个
        
        
        print("######### RANK LIST ########")
        for i, (entity, pred_value) in enumerate(pairs_sorted[:2000]):
            print(f"RANK: {i+1}, 尾实体: {entity_vocab[entity]}, 预测值: {pred_value}")
        
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1
        # rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
        return pairs_sorted
    
def visualize_raw(solver, sample, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    h_index, t_index, r_index = sample.unbind(-1) # 分离头实体、尾实体和关系
    inverse = torch.stack([t_index, h_index, r_index + num_relation], dim=-1) # 生成反向三元组
    batch = sample.unsqueeze(0) # 将样本扩展为批次
    if sample.ndim == 1: # 如果样本是一维的，则将其扩展为批次
        vis_batch = torch.stack([sample, inverse]) # 生成正向和反向三元组的批次
    else:
        is_t_neg = (h_index == h_index[0]).all() # 判断是否为尾实体负例
        vis_batch = sample[:1] if is_t_neg else inverse[:1] # 如果是尾实体负例，则选择正向三元组，否则选择反向三元组
    batch = batch.to(solver.device) 
    vis_batch = vis_batch.to(solver.device)

    solver.model.eval()

    with torch.no_grad():
        pred, target = solver.model.predict_and_target(batch) # 预测和目标
    if isinstance(target, tuple):
        mask, target = target # 如果目标是一个元组，则分离掩码和目标
        pos_pred = pred.gather(-1, target.unsqueeze(-1)) # 获取正例预测值
        rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
    else:
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1
        
    # print(">>>>>>>>>>>>>>pos_pred>>>>>>>>>>>>>>")
    # print(pos_pred)
    paths, weights, num_steps = solver.model.visualize(vis_batch) # 可视化路径
    batch = batch.tolist()
    rankings = rankings.tolist()
    paths = paths.tolist()
    weights = weights.tolist()
    num_steps = num_steps.tolist()

    logger.warning("")
    for i in range(len(vis_batch)):
        h, t, r = vis_batch[i] # 获取可视化批次中的头实体、尾实体和关系
        h_token = entity_vocab[h]
        t_token = entity_vocab[t]
        r_token = relation_vocab[r % num_relation]
        if r >= num_relation: # 如果关系是反向关系，则添加"^(-1)"
            r_token += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_token, h_token, r_token, rankings[i]))

        for path, weight, num_step in zip(paths[i], weights[i], num_steps[i]): # 遍历路径、权重和步数
            if weight == float("-inf"): # 如果权重为负无穷，则跳过
                break
            triplets = []
            for h, t, r in path[:num_step]:
                h_token = entity_vocab[h]
                t_token = entity_vocab[t]
                r_token = relation_vocab[r % num_relation]
                if r >= num_relation:
                    r_token += "^(-1)"
                triplets.append("<%s, %s, %s>" % (h_token, r_token, t_token))
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))


def read_train_data(file_path):
    # 初始化二维列表
    data = []
    try:
        # 打开文件并按行读取
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除每行首尾的空白字符（包括换行符）
                line = line.strip()
                if line:  # 跳过空行
                    # 按制表符分割为三列，存入子列表
                    columns = line.split('\t')
                    # 确保每行确实有三列数据
                    if len(columns) == 3:
                        data.append(columns)
                    else:
                        print(f"警告：行'{line}'不符合三列格式，已跳过")
        print(f"成功读取 {len(data)} 行数据")
        return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None







if __name__ == "__main__":
    print("======优化了main函数=====")
    print("======优化了main函数=====")
    print("======优化了main函数=====")
    
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    
    # ---------------------------------------------------------
    # 1. 模型与词表加载
    # ---------------------------------------------------------
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    entity_vocab, relation_vocab = load_vocab(dataset)

    h_entitis = [1230427, 88092, 368611, 208341, 182428, 94518, 51948, 38659,
                 111357, 40348, 112448, 25637, 37855, 28586, 66818, 16078, 58473, 
                 452588, 3380, 58591, 176295, 18181, 20759, 21513, 37855, 150737, 
                 618638, 25472, 90773, 67668, 3710, 12592, 177278, 6981, 104144, 
                 205002, 170012, 49031, 26084, 13994, 19023, 20154, 44572, 361607, 
                 13279, 85417, 314765, 1309, 5973, 162969, 21204, 87034, 289569, 
                 253283, 230884, 3267, 779082, 402750, 2535688]

    # ---------------------------------------------------------
    # 2. 数据预处理（移出循环，防止重复计算和内存溢出）
    # ---------------------------------------------------------
    file_path = "/dataStor/home/yqzhang/data/mic_data_0523/train.txt"
    train_data = read_train_data(file_path)

    print("正在构建训练集索引 (Set构建)...")
    train_triple_set = set()
    if train_data:
        for sample in train_data:
            # 确保存入的是 tuple，且数据类型一致（通常是字符串）
            train_triple_set.add((sample[0], sample[1], sample[2]))
    print(f"训练集索引构建完成，共 {len(train_triple_set)} 条唯一记录。")

    # ---------------------------------------------------------
    # 3. 预先提取目标实体索引（移出循环）
    # ---------------------------------------------------------
    r_entity = 52 + 156
    t_class = "Microbe"
    print(f"正在筛选类型为 {t_class} 的尾实体...")
    t_entities = []
    for entity_idx, entity_str in enumerate(entity_vocab):
        if entity_str.startswith(t_class + "//"):
            t_entities.append(entity_idx)
    print(f"筛选完成，共找到 {len(t_entities)} 个目标实体。")

    # ---------------------------------------------------------
    # 4. 开始预测循环
    # ---------------------------------------------------------
    solver.model.eval()
    
    # 只需要移动一次到 GPU，避免在循环内反复操作（如果是固定 tensor）
    # 但由于 batch 是动态构建的，我们在循环内处理
    
    for idx, h_entity in enumerate(h_entitis):
        print(f"\nProcessing head entity {idx+1}/{len(h_entitis)}: {entity_vocab[h_entity]} (ID: {h_entity})")
        
        # 显式进行无梯度上下文，虽然函数里有，这里加一层更保险
        with torch.no_grad():
            # 构造查询张量
            query_h_t_r = torch.tensor([h_entity, 66100, r_entity])
            
            # 获取排序结果
            pairs_sorted = rank(solver, query_h_t_r, entity_vocab, relation_vocab, t_entities)
            
            display_count = 0
            max_display = 1
            
            if pairs_sorted:
                for rank_idx, (tail_entity_idx, pred_value) in enumerate(pairs_sorted[:300]):
                    head_name = entity_vocab[h_entity]
                    relation_name = relation_vocab[r_entity]
                    tail_name = entity_vocab[tail_entity_idx]
                    
                    # 检查是否在训练集中，保证是新发现的关系
                    if (head_name, relation_name, tail_name) not in train_triple_set:
                        print(f"RANK: {rank_idx + 1}")
                        print(f"尾实体: {tail_name} (索引: {tail_entity_idx})")
                        print(f"查询三元组: ({head_name}, {relation_name}, {tail_name})")
                        print(f"预测值: {pred_value}\n")
                        
                        print("=================CHANGE_RICHNESS_DcrMi==================")
                        # 构造可视化用的 Tensor
                        vis_input = torch.tensor([h_entity, tail_entity_idx, r_entity])
                        
                        visualize_raw(
                            solver,
                            vis_input,
                            entity_vocab,
                            relation_vocab
                        )
                        print("--------------------------------------------------------\n")
                        
                        # 显式删除临时 Tensor，帮助释放显存
                        del vis_input
                        
                        display_count += 1
                        if display_count >= max_display:
                            break
            
            # 显式删除本轮循环的大对象
            del pairs_sorted
            del query_h_t_r
            
            # 【关键】每处理完一个头实体，清理一次显存缓存
            # 注意：频繁调用 empty_cache 会稍微降低速度，但能有效防止显存虚高
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("全部处理完成。")
