import os
import csv
import glob
from tqdm import tqdm
from ogb import linkproppred

import torch
from torch.utils import data as torch_data

from reasoning.TorchDrug import data, datasets, utils
from reasoning.TorchDrug.core import Registry as R


# class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

#     def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
#         assert len(transductive_files) == len(inductive_files) == 3 #每一行三个元素
#         inv_transductive_vocab = {} #头实体（h）和尾实体（t）被存储在这个字典中，映射为一个唯一的整数 ID
#         inv_inductive_vocab = {} 
#         inv_relation_vocab = {} #关系（r）被存储在这个字典中，映射为一个唯一的整数 ID
#         triplets = []
#         num_samples = []

#         for txt_file in transductive_files:
#             with open(txt_file, "r") as fin:
#                 reader = csv.reader(fin, delimiter="\t")
#                 if verbose:
#                     reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

#                 num_sample = 0
#                 for tokens in reader:
#                     h_token, r_token, t_token = tokens #读取头实体、关系和尾实体
#                     if h_token not in inv_transductive_vocab:
#                         inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
#                     h = inv_transductive_vocab[h_token]
#                     if r_token not in inv_relation_vocab:
#                         inv_relation_vocab[r_token] = len(inv_relation_vocab)
#                     r = inv_relation_vocab[r_token]
#                     if t_token not in inv_transductive_vocab:
#                         inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
#                     t = inv_transductive_vocab[t_token]
#                     triplets.append((h, t, r)) # 处理完的三元组（(h, t, r)）被存储在 triplets 列表中
#                     num_sample += 1
#             num_samples.append(num_sample)

#         for txt_file in inductive_files:
#             with open(txt_file, "r") as fin:
#                 reader = csv.reader(fin, delimiter="\t")
#                 if verbose:
#                     reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

#                 num_sample = 0
#                 for tokens in reader:
#                     h_token, r_token, t_token = tokens
#                     if h_token not in inv_inductive_vocab:
#                         inv_inductive_vocab[h_token] = len(inv_inductive_vocab)
#                     h = inv_inductive_vocab[h_token]
#                     assert r_token in inv_relation_vocab
#                     r = inv_relation_vocab[r_token]
#                     if t_token not in inv_inductive_vocab:
#                         inv_inductive_vocab[t_token] = len(inv_inductive_vocab)
#                     t = inv_inductive_vocab[t_token]
#                     triplets.append((h, t, r)) # 归纳式数据中的实体被存储在inv_inductive_vocab字典中，与传导式数据分开。
#                     num_sample += 1
#             num_samples.append(num_sample)

#         transductive_vocab, inv_transductive_vocab = self._standarize_vocab(None, inv_transductive_vocab) # 标准化传导式数据中的实体
#         inductive_vocab, inv_inductive_vocab = self._standarize_vocab(None, inv_inductive_vocab)
#         relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab) # 标准化关系

#         # fact_graph: 表示传导式知识图谱，包含传导式数据中的事实。
#         self.fact_graph = data.Graph(triplets[:num_samples[0]],
#                                      num_node=len(transductive_vocab), num_relation=len(relation_vocab))
#         # graph: 表示整个传导式图
#         self.graph = data.Graph(triplets[:sum(num_samples[:3])],
#                                 num_node=len(transductive_vocab), num_relation=len(relation_vocab))
#         # transductive_fact_graph: 表示归纳式知识图谱中的事实。
#         self.inductive_fact_graph = data.Graph(triplets[sum(num_samples[:3]): sum(num_samples[:4])],
#                                                num_node=len(inductive_vocab), num_relation=len(relation_vocab))
#         # inductive_graph: 表示整个归纳式图。
#         self.inductive_graph = data.Graph(triplets[sum(num_samples[:3]):],
#                                           num_node=len(inductive_vocab), num_relation=len(relation_vocab))
#         # 将所有处理后的三元组转换为 PyTorch 张量存储在 self.triplets 中，后续可以通过索引来获取。
#         self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:4]):])
#         self.num_samples = num_samples[:2] + [sum(num_samples[4:])]
#         self.transductive_vocab = transductive_vocab
#         self.inductive_vocab = inductive_vocab
#         self.relation_vocab = relation_vocab
#         self.inv_transductive_vocab = inv_transductive_vocab
#         self.inv_inductive_vocab = inv_inductive_vocab
#         self.inv_relation_vocab = inv_relation_vocab

#     def __getitem__(self, index):
#         return self.triplets[index]

#     def split(self):
#         offset = 0
#         splits = []
#         for num_sample in self.num_samples:
#             split = torch_data.Subset(self, range(offset, offset + num_sample))
#             splits.append(split)
#             offset += num_sample
#         return splits

class TransductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    def load_transductive_tsvs(self, transductive_files, verbose=0):
        assert len(transductive_files) == 3  # 确保有三个文件
        inv_transductive_vocab = {} 
        inv_relation_vocab = {} 
        triplets = []
        num_samples = []

        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_transductive_vocab:
                        inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
                    h = inv_transductive_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_transductive_vocab:
                        inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
                    t = inv_transductive_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        transductive_vocab, inv_transductive_vocab = self._standarize_vocab(None, inv_transductive_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        # 这里只加载传导式数据，不涉及归纳式数据
        self.fact_graph = data.Graph(triplets[:num_samples[0]], 
                                     num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        self.graph = data.Graph(triplets[:sum(num_samples[:3])], 
                                num_node=len(transductive_vocab), num_relation=len(relation_vocab))

        self.triplets = torch.tensor(triplets[:sum(num_samples)])  # 使用所有的传导式三元组
        self.num_samples = num_samples
        self.transductive_vocab = transductive_vocab
        self.relation_vocab = relation_vocab
        self.inv_transductive_vocab = inv_transductive_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("dataset.MicrobeKGTransductive")
class MicrobeKGTransductive(TransductiveKnowledgeGraphDataset):
    
    transductive_urls = [
        "/public/home/yqzhang/modeltest/0_path_model/AStarNet-master/datasets/knowledge_graphs/microbekg_train.txt",
        "/public/home/yqzhang/modeltest/0_path_model/AStarNet-master/datasets/knowledge_graphs/microbekg_valid.txt",
        "/public/home/yqzhang/modeltest/0_path_model/AStarNet-master/datasets/knowledge_graphs/microbekg_test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = self.transductive_urls  

        self.load_transductive_tsvs(transductive_files, verbose=verbose)

# class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    # transductive_urls = [
    #     "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
    #     "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
    #     "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    # ]

    # inductive_urls = [
    #     "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
    #     "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
    #     "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    # ]

    # def __init__(self, path, version="v1", verbose=1):
    #     path = os.path.expanduser(path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     self.path = path

    #     transductive_files = []
    #     for url in self.transductive_urls:
    #         url = url % version
    #         save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
    #         txt_file = os.path.join(path, save_file)
    #         if not os.path.exists(txt_file):
    #             txt_file = utils.download(url, self.path, save_file=save_file)
    #         transductive_files.append(txt_file)
    #     inductive_files = []
    #     for url in self.inductive_urls:
    #         url = url % version
    #         save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
    #         txt_file = os.path.join(path, save_file)
    #         if not os.path.exists(txt_file):
    #             txt_file = utils.download(url, self.path, save_file=save_file)
    #         inductive_files.append(txt_file)

    #     self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


# @R.register("dataset.WN18RRInductive")
# class WN18RRInductive(InductiveKnowledgeGraphDataset):

#     transductive_urls = [
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
#     ]

#     inductive_urls = [
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
#     ]

#     def __init__(self, path, version="v1", verbose=1):
#         path = os.path.expanduser(path)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         self.path = path

#         transductive_files = []
#         for url in self.transductive_urls:
#             url = url % version
#             save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
#             txt_file = os.path.join(path, save_file)
#             if not os.path.exists(txt_file):
#                 txt_file = utils.download(url, self.path, save_file=save_file)
#             transductive_files.append(txt_file)
#         inductive_files = []
#         for url in self.inductive_urls:
#             url = url % version
#             save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
#             txt_file = os.path.join(path, save_file)
#             if not os.path.exists(txt_file):
#                 txt_file = utils.download(url, self.path, save_file=save_file)
#             inductive_files.append(txt_file)

#         self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


# @R.register("dataset.OGBLWikiKG2") # 这是一个装饰器，用于注册这个类，方便引用
# class OGBLWikiKG2(data.KnowledgeGraphDataset):
#     """
#     OGBLWikiKG2(
#         #entity: 2,500,604
#         #relation: 535
#         #triplet: 17,137,181
#     )
#     #train: 16,109,182, #valid: 858,912, #test: 1,197,086
#     """

#     def __init__(self, path, verbose=1):
#         path = os.path.expanduser(path)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         self.path = path

#         dataset = linkproppred.LinkPropPredDataset("ogbl-wikikg2", path) # 使用 OGB-LINKPROPPRED 库加载 WikiKG2 数据集
#         self.load_ogb(dataset, verbose=verbose) # 加载 OGB 数据集

#     def load_ogb(self, dataset, verbose=1): # 加载 OGB 数据集
#         inv_entity_vocab = {} 
#         inv_relation_vocab = {} # 创建一个空字典，用于存储实体和关系的映射

#         zip_files = glob.glob(os.path.join(dataset.root, "mapping/*.csv.gz")) 
#         # 从mapping目录中加载实体和关系的映射文件
#         for zip_file in zip_files:
#             csv_file = utils.extract(zip_file) # 解压csv文件
#             with open(csv_file, "r") as fin:
#                 reader = csv.reader(fin) # 打开csv文件
#                 if verbose:
#                     reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
#                 fields = next(reader)
#                 if "reltype" in csv_file:
#                     for index, token in reader:
#                         inv_relation_vocab[token] = int(index)
#                 elif "nodeidx" in csv_file:
#                     for index, token in reader:
#                         inv_entity_vocab[token] = int(index)
#                 else:
#                     raise RuntimeError("Unknown mapping file `%s`" % csv_file)

#         edge_split = dataset.get_edge_split() # 获取训练集、验证集和测试集的边信息
#         triplets = []
#         num_samples = []
#         num_samples_with_neg = []
#         negative_heads = []
#         negative_tails = []
#         for key in ["train", "valid", "test"]:
#             split_dict = edge_split[key]
#             h = torch.as_tensor(split_dict["head"])
#             t = torch.as_tensor(split_dict["tail"])
#             r = torch.as_tensor(split_dict["relation"])

#             triplet = torch.stack([h, t, r], dim=-1)
#             triplets.append(triplet)
#             num_samples.append(len(triplet))
#             if "head_neg" in split_dict: # 如果存在负样本，则将负样本添加到列表中
#                 neg_h = torch.as_tensor(split_dict["head_neg"])
#                 neg_t = torch.as_tensor(split_dict["tail_neg"])
#                 negative_heads.append(neg_h) 
#                 negative_tails.append(neg_t) # 将负样本添加到列表中
#                 num_samples_with_neg.append(len(neg_h))
#             else:
#                 num_samples_with_neg.append(0)
#         triplets = torch.cat(triplets)

#         self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
#         self.num_samples = num_samples
#         self.num_samples_with_neg = num_samples_with_neg
#         self.negative_heads = torch.cat(negative_heads)
#         self.negative_tails = torch.cat(negative_tails)
#         self.name = dataset.name

#     def split(self, test_negative=True): # 将数据集划分为训练集、验证集和测试集
#         offset = 0
#         neg_offset = 0 # offset 和 neg_offset 用于跟踪当前处理的样本和负样本的偏移量
#         splits = []
#         for num_sample, num_sample_with_neg in zip(self.num_samples, self.num_samples_with_neg):
#             if test_negative and num_sample_with_neg:
#                 triplets = self[offset: offset + num_sample]
#                 negative_heads = self.negative_heads[neg_offset: neg_offset + num_sample_with_neg]
#                 negative_tails = self.negative_tails[neg_offset: neg_offset + num_sample_with_neg]
#                 split = OGBLKGTest(triplets, negative_heads, negative_tails)
#             else:
#                 split = torch_data.Subset(self, range(offset, offset + num_sample))
#                 # 根据是否有负样本，将数据集分割成不同的部分，并存储在 splits 列表中
#             splits.append(split)
#             offset += num_sample
#             neg_offset += num_sample_with_neg
#         return splits


# class OGBLKGTest(torch_data.Dataset): # OGBLKGTest 类用于处理 OGB-LSC 数据集的测试集

#     def __init__(self, triplets, negative_heads, negative_tails):
#         self.triplets = triplets
#         self.negative_heads = negative_heads
#         self.negative_tails = negative_tails
#         self.num_negative = negative_heads.shape[-1]

#     def __getitem__(self, index): # 根据索引返回三元组及其对应的负样本
#         assert isinstance(index, int)

#         is_t_neg = index // len(self.triplets) == 0
#         index = index % len(self.triplets)
#         triplet = self.triplets[index]
#         triplet = triplet.repeat(self.num_negative + 1, 1)
#         if is_t_neg:
#             triplet[1:, 1] = self.negative_tails[index]
#         else:
#             triplet[1:, 0] = self.negative_heads[index]
#         return triplet

#     def __len__(self):
#         return len(self.triplets) * 2 
#     # 返回测试集中的总样本数，包括正样本和负样本。