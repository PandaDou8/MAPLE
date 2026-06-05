import os
import csv
import glob
from tqdm import tqdm
from ogb import linkproppred

import torch
from torch.utils import data as torch_data

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reasoning.TorchDrug import data, datasets, utils, core
from reasoning.TorchDrug.core import Registry as R
from config import config


class KnowledgeGraphDataset(torch_data.Dataset, core.Configurable):
    """
    Knowledge graph dataset.

    The whole dataset contains one knowledge graph.
    """

    def load_triplet(self, triplets, entity_vocab=None, relation_vocab=None, inv_entity_vocab=None,
                     inv_relation_vocab=None):
        """
        Load the dataset from triplets.
        The mapping between indexes and tokens is specified through either vocabularies or inverse vocabularies.

        Parameters:
            triplets (array_like): triplets of shape :math:`(n, 3)`
            entity_vocab (dict of str, optional): maps entity indexes to tokens
            relation_vocab (dict of str, optional): maps relation indexes to tokens
            inv_entity_vocab (dict of str, optional): maps tokens to entity indexes
            inv_relation_vocab (dict of str, optional): maps tokens to relation indexes
        """
        entity_vocab, inv_entity_vocab = self._standarize_vocab(entity_vocab, inv_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, inv_relation_vocab)

        num_node = len(entity_vocab) if entity_vocab else None
        num_relation = len(relation_vocab) if relation_vocab else None
        self.graph = data.Graph(triplets, num_node=num_node, num_relation=num_relation)
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_entity_vocab = inv_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def load_tsv(self, tsv_file, verbose=0):
        """
        Load the dataset from a tsv file.

        Parameters:
            tsv_file (str): file name
            verbose (int, optional): output verbose level
        """
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []

        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = tqdm(reader, "Loading %s" % tsv_file)
            for tokens in reader:
                h_token, r_token, t_token = tokens
                if h_token not in inv_entity_vocab:
                    inv_entity_vocab[h_token] = len(inv_entity_vocab)
                h = inv_entity_vocab[h_token]
                if r_token not in inv_relation_vocab:
                    inv_relation_vocab[r_token] = len(inv_relation_vocab)
                r = inv_relation_vocab[r_token]
                if t_token not in inv_entity_vocab:
                    inv_entity_vocab[t_token] = len(inv_entity_vocab)
                t = inv_entity_vocab[t_token]
                triplets.append((h, t, r))

        self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)

    def load_tsvs(self, tsv_files, verbose=0, inv_entity_vocab=None, inv_relation_vocab=None):
        """
        Load the dataset from multiple tsv files.

        Parameters:
            tsv_files (list of str): list of file names
            verbose (int, optional): output verbose level
        """
        inv_entity_vocab = dict(inv_entity_vocab or {})
        inv_relation_vocab = dict(inv_relation_vocab or {})
        triplets = []
        num_samples = []

        for tsv_file in tsv_files:
            with open(tsv_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % tsv_file, utils.get_line_count(tsv_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_entity_vocab:
                        inv_entity_vocab[h_token] = len(inv_entity_vocab)
                    h = inv_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_entity_vocab:
                        inv_entity_vocab[t_token] = len(inv_entity_vocab)
                    t = inv_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
        self.num_samples = num_samples

    def _standarize_vocab(self, vocab, inverse_vocab):
        if vocab is not None:
            if isinstance(vocab, dict):
                assert set(vocab.keys()) == set(range(len(vocab))), "Vocab keys should be consecutive numbers"
                vocab = [vocab[k] for k in range(len(vocab))]
            if inverse_vocab is None:
                inverse_vocab = {v: i for i, v in enumerate(vocab)}
        if inverse_vocab is not None:
            assert set(inverse_vocab.values()) == set(range(len(inverse_vocab))), \
                "Inverse vocab values should be consecutive numbers"
            if vocab is None:
                vocab = sorted(inverse_vocab, key=lambda k: inverse_vocab[k])
        return vocab, inverse_vocab

    @property
    def num_entity(self):
        """Number of entities."""
        return self.graph.num_node

    @property
    def num_triplet(self):
        """Number of triplets."""
        return self.graph.num_edge

    @property
    def num_relation(self):
        """Number of relations."""
        return self.graph.num_relation

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def __repr__(self):
        lines = [
            "#entity: %d" % self.num_entity,
            "#relation: %d" % self.num_relation,
            "#triplet: %d" % self.num_triplet,
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

# class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):
class InductiveKnowledgeGraphDataset(KnowledgeGraphDataset):

    def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
        assert len(transductive_files) == len(inductive_files) == 3 # Each file group has three splits.
        inv_transductive_vocab = {} # Map transductive entity tokens to integer IDs.
        inv_inductive_vocab = {} 
        inv_relation_vocab = {} # Map relation tokens to integer IDs.
        triplets = []
        num_samples = []

        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens # Read head, relation, and tail tokens.
                    if h_token not in inv_transductive_vocab:
                        inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
                    h = inv_transductive_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_transductive_vocab:
                        inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
                    t = inv_transductive_vocab[t_token]
                    triplets.append((h, t, r)) # Store processed triples as (head, tail, relation).
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in inductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_inductive_vocab:
                        inv_inductive_vocab[h_token] = len(inv_inductive_vocab)
                    h = inv_inductive_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_inductive_vocab:
                        inv_inductive_vocab[t_token] = len(inv_inductive_vocab)
                    t = inv_inductive_vocab[t_token]
                    triplets.append((h, t, r)) # Store inductive entities separately from transductive entities.
                    num_sample += 1
            num_samples.append(num_sample)

        transductive_vocab, inv_transductive_vocab = self._standarize_vocab(None, inv_transductive_vocab) # Standardize transductive entity vocabularies.
        inductive_vocab, inv_inductive_vocab = self._standarize_vocab(None, inv_inductive_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab) # Standardize relation vocabularies.

        # fact_graph contains the transductive training facts.
        self.fact_graph = data.Graph(triplets[:num_samples[0]],
                                     num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        # graph contains the full transductive graph.
        self.graph = data.Graph(triplets[:sum(num_samples[:3])],
                                num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        # transductive_fact_graph stores training facts for transductive entities.
        self.inductive_fact_graph = data.Graph(triplets[sum(num_samples[:3]): sum(num_samples[:4])],
                                               num_node=len(inductive_vocab), num_relation=len(relation_vocab))
        # inductive_graph contains the full inductive graph.
        self.inductive_graph = data.Graph(triplets[sum(num_samples[:3]):],
                                          num_node=len(inductive_vocab), num_relation=len(relation_vocab))
        # Store processed triples as tensors for indexed access.
        self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:4]):])
        self.num_samples = num_samples[:2] + [sum(num_samples[4:])]
        self.transductive_vocab = transductive_vocab
        self.inductive_vocab = inductive_vocab
        self.relation_vocab = relation_vocab
        self.inv_transductive_vocab = inv_transductive_vocab
        self.inv_inductive_vocab = inv_inductive_vocab
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
class MicrobeKGTransductive(KnowledgeGraphDataset):
    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        transductive_files = self._get_transductive_files()
        cache_file = self._get_cache_file()

        # Prefer local cache for large graphs to avoid rescanning huge TSV files.
        # Build the cache on first use; reuse it while source files stay unchanged.
        if self._cache_is_fresh(cache_file, transductive_files):
            self._load_cache(cache_file)
        else:
            inv_entity_vocab, inv_relation_vocab = self._load_mapping_vocab()
            self.load_tsvs(
                transductive_files,
                verbose=verbose,
                inv_entity_vocab=inv_entity_vocab,
                inv_relation_vocab=inv_relation_vocab,
            )
            self._save_cache(cache_file)

    def _get_transductive_files(self):
        return [
            os.path.join(self.path, "train.txt"),
            os.path.join(self.path, "valid.txt"),
            os.path.join(self.path, "test.txt"),
        ]

    def _get_cache_file(self):
        return os.path.join(self.path, "microbekg_transductive_cache.pt")

    def _load_mapping_vocab(self):
        entity_vocab = self._read_mapping_file("entity")
        relation_vocab = self._read_mapping_file("relation")
        return entity_vocab, relation_vocab

    def _read_mapping_file(self, name):
        candidates = [
            os.path.join(self.path, "%s.txt" % name),
            os.path.join(self.path, "mappings", "%s.txt" % name),
        ]
        mapping_file = next((path for path in candidates if os.path.exists(path)), None)
        if mapping_file is None:
            return None

        mapping = {}
        with open(mapping_file, "r", encoding="utf-8") as fin:
            for line in fin:
                token, index = line.rstrip("\n").split("\t")
                mapping[token] = int(index)
        return mapping

    def _cache_is_fresh(self, cache_file, source_files):
        if not os.path.exists(cache_file):
            return False
        cache_mtime = os.path.getmtime(cache_file)
        return all(os.path.exists(source_file) and os.path.getmtime(source_file) <= cache_mtime
                   for source_file in source_files)

    def _load_cache(self, cache_file):
        state = torch.load(cache_file, map_location="cpu")
        self.load_triplet(
            state["triplets"],
            entity_vocab=state.get("entity_vocab"),
            relation_vocab=state.get("relation_vocab"),
        )
        self.num_samples = state["num_samples"]

    def _save_cache(self, cache_file):
        state = {
            "triplets": self.graph.edge_list.cpu(),
            "num_samples": self.num_samples,
            "entity_vocab": self.entity_vocab,
            "relation_vocab": self.relation_vocab,
        }
        tmp_cache_file = cache_file + ".tmp"
        torch.save(state, tmp_cache_file)
        os.replace(tmp_cache_file, cache_file)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


# @R.register("datasets.FB15k237")
# class FB15k237(KnowledgeGraphDataset):
#     """
#     A filtered version of FB15k dataset without trivial cases.

#     Statistics:
#         - #Entity: 14,541
#         - #Relation: 237
#         - #Triplet: 310,116

#     Parameters:
#         path (str): path to store the dataset
#         verbose (int, optional): output verbose level
#     """

#     urls = [
#         "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/train.txt",
#         "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/valid.txt",
#         "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/test.txt",
#     ]
#     md5s = [
#         "c05b87b9ac00f41901e016a2092d7837",
#         "6a94efd530e5f43fcf84f50bc6d37b69",
#         "f5bdf63db39f455dec0ed259bb6f8628"
#     ]

#     def __init__(self, path, verbose=1):
#         path = os.path.expanduser(path)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         self.path = path

#         txt_files = []
#         for url, md5 in zip(self.urls, self.md5s):
#             save_file = "fb15k237_%s" % os.path.basename(url)
#             txt_file = utils.download(url, self.path, save_file=save_file, md5=md5)
#             txt_files.append(txt_file)

#         self.load_tsvs(txt_files, verbose=verbose)

#     def split(self):
#         offset = 0
#         splits = []
#         for num_sample in self.num_samples:
#             split = torch_data.Subset(self, range(offset, offset + num_sample))
#             splits.append(split)
#             offset += num_sample
#         return splits

@R.register("dataset.FB15k237Inductive")
class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


@R.register("datasets.WN18RR")
class WN18RR(KnowledgeGraphDataset):
    """
    A filtered version of WN18 dataset without trivial cases.

    Statistics:
        - #Entity: 40,943
        - #Relation: 11
        - #Triplet: 93,003

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/test.txt",
    ]
    md5s = [
        "35e81af3ae233327c52a87f23b30ad3c",
        "74a2ee9eca9a8d31f1a7d4d95b5e0887",
        "2b45ba1ba436b9d4ff27f1d3511224c9"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "wn18rr_%s" % os.path.basename(url)
            txt_file = utils.download(url, self.path, save_file=save_file, md5=md5)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

@R.register("dataset.WN18RRInductive")
class WN18RRInductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


@R.register("dataset.OGBLWikiKG2") # Register this dataset class.
class OGBLWikiKG2(KnowledgeGraphDataset):
    """
    OGBLWikiKG2(
        #entity: 2,500,604
        #relation: 535
        #triplet: 17,137,181
    )
    #train: 16,109,182, #valid: 858,912, #test: 1,197,086
    """

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        dataset = linkproppred.LinkPropPredDataset("ogbl-wikikg2", path) # Load the WikiKG2 dataset through OGB LinkPropPred.
        self.load_ogb(dataset, verbose=verbose) # Load the OGB dataset.

    def load_ogb(self, dataset, verbose=1): # Load the OGB dataset.
        inv_entity_vocab = {} 
        inv_relation_vocab = {} # Store entity and relation mappings.

        zip_files = glob.glob(os.path.join(dataset.root, "mapping/*.csv.gz")) 
        # Load entity and relation mapping files.
        for zip_file in zip_files:
            csv_file = utils.extract(zip_file) # Extract the CSV file.
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin) # Read the CSV file.
                if verbose:
                    reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
                fields = next(reader)
                if "reltype" in csv_file:
                    for index, token in reader:
                        inv_relation_vocab[token] = int(index)
                elif "nodeidx" in csv_file:
                    for index, token in reader:
                        inv_entity_vocab[token] = int(index)
                else:
                    raise RuntimeError("Unknown mapping file `%s`" % csv_file)

        edge_split = dataset.get_edge_split() # Read train / valid / test edge splits.
        triplets = []
        num_samples = []
        num_samples_with_neg = []
        negative_heads = []
        negative_tails = []
        for key in ["train", "valid", "test"]:
            split_dict = edge_split[key]
            h = torch.as_tensor(split_dict["head"])
            t = torch.as_tensor(split_dict["tail"])
            r = torch.as_tensor(split_dict["relation"])

            triplet = torch.stack([h, t, r], dim=-1)
            triplets.append(triplet)
            num_samples.append(len(triplet))
            if "head_neg" in split_dict: # Add provided negative samples when available.
                neg_h = torch.as_tensor(split_dict["head_neg"])
                neg_t = torch.as_tensor(split_dict["tail_neg"])
                negative_heads.append(neg_h) 
                negative_tails.append(neg_t) # Add negative tails.
                num_samples_with_neg.append(len(neg_h))
            else:
                num_samples_with_neg.append(0)
        triplets = torch.cat(triplets)

        self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
        self.num_samples = num_samples
        self.num_samples_with_neg = num_samples_with_neg
        self.negative_heads = torch.cat(negative_heads)
        self.negative_tails = torch.cat(negative_tails)
        self.name = dataset.name

    def split(self, test_negative=True): # Split the dataset into train / valid / test sets.
        offset = 0
        neg_offset = 0 # Track positive and negative sample offsets.
        splits = []
        for num_sample, num_sample_with_neg in zip(self.num_samples, self.num_samples_with_neg):
            if test_negative and num_sample_with_neg:
                triplets = self[offset: offset + num_sample]
                negative_heads = self.negative_heads[neg_offset: neg_offset + num_sample_with_neg]
                negative_tails = self.negative_tails[neg_offset: neg_offset + num_sample_with_neg]
                split = OGBLKGTest(triplets, negative_heads, negative_tails)
            else:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                # Store each split according to negative-sample availability.
            splits.append(split)
            offset += num_sample
            neg_offset += num_sample_with_neg
        return splits


class OGBLKGTest(torch_data.Dataset): # Test split wrapper for OGB-LSC datasets.

    def __init__(self, triplets, negative_heads, negative_tails):
        self.triplets = triplets
        self.negative_heads = negative_heads
        self.negative_tails = negative_tails
        self.num_negative = negative_heads.shape[-1]

    def __getitem__(self, index): # Return a triple and its corresponding negative samples.
        assert isinstance(index, int)

        is_t_neg = index // len(self.triplets) == 0
        index = index % len(self.triplets)
        triplet = self.triplets[index]
        triplet = triplet.repeat(self.num_negative + 1, 1)
        if is_t_neg:
            triplet[1:, 1] = self.negative_tails[index]
        else:
            triplet[1:, 0] = self.negative_heads[index]
        return triplet

    def __len__(self):
        return len(self.triplets) * 2 
    # Return the total number of positive and negative test samples.


# import os
# import csv
# import glob
# from tqdm import tqdm
# from ogb import linkproppred

# import torch
# from torch.utils import data as torch_data


# from reasoning.TorchDrug import data, datasets, utils
# from reasoning.TorchDrug.core import Registry as R


# class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

#     def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
#         assert len(transductive_files) == len(inductive_files) == 3
#         inv_transductive_vocab = {}
#         inv_inductive_vocab = {}
#         inv_relation_vocab = {}
#         triplets = []
#         num_samples = []

#         for txt_file in transductive_files:
#             with open(txt_file, "r") as fin:
#                 reader = csv.reader(fin, delimiter="\t")
#                 if verbose:
#                     reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

#                 num_sample = 0
#                 for tokens in reader:
#                     h_token, r_token, t_token = tokens
#                     if h_token not in inv_transductive_vocab:
#                         inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
#                     h = inv_transductive_vocab[h_token]
#                     if r_token not in inv_relation_vocab:
#                         inv_relation_vocab[r_token] = len(inv_relation_vocab)
#                     r = inv_relation_vocab[r_token]
#                     if t_token not in inv_transductive_vocab:
#                         inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
#                     t = inv_transductive_vocab[t_token]
#                     triplets.append((h, t, r))
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
#                     triplets.append((h, t, r))
#                     num_sample += 1
#             num_samples.append(num_sample)

#         transductive_vocab, inv_transductive_vocab = self._standarize_vocab(None, inv_transductive_vocab)
#         inductive_vocab, inv_inductive_vocab = self._standarize_vocab(None, inv_inductive_vocab)
#         relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

#         self.fact_graph = data.Graph(triplets[:num_samples[0]],
#                                      num_node=len(transductive_vocab), num_relation=len(relation_vocab))
#         self.graph = data.Graph(triplets[:sum(num_samples[:3])],
#                                 num_node=len(transductive_vocab), num_relation=len(relation_vocab))
#         self.inductive_fact_graph = data.Graph(triplets[sum(num_samples[:3]): sum(num_samples[:4])],
#                                                num_node=len(inductive_vocab), num_relation=len(relation_vocab))
#         self.inductive_graph = data.Graph(triplets[sum(num_samples[:3]):],
#                                           num_node=len(inductive_vocab), num_relation=len(relation_vocab))
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


# @R.register("dataset.FB15k237Inductive")
# class FB15k237Inductive(InductiveKnowledgeGraphDataset):

#     transductive_urls = [
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
#     ]

#     inductive_urls = [
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
#         "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
#     ]

#     def __init__(self, path, version="v1", verbose=1):
#         path = os.path.expanduser(path)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         self.path = path

#         transductive_files = []
#         for url in self.transductive_urls:
#             url = url % version
#             save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
#             txt_file = os.path.join(path, save_file)
#             if not os.path.exists(txt_file):
#                 txt_file = utils.download(url, self.path, save_file=save_file)
#             transductive_files.append(txt_file)
#         inductive_files = []
#         for url in self.inductive_urls:
#             url = url % version
#             save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
#             txt_file = os.path.join(path, save_file)
#             if not os.path.exists(txt_file):
#                 txt_file = utils.download(url, self.path, save_file=save_file)
#             inductive_files.append(txt_file)

#         self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


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


# @R.register("dataset.OGBLWikiKG2")
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

#         dataset = linkproppred.LinkPropPredDataset("ogbl-wikikg2", path)
#         self.load_ogb(dataset, verbose=verbose)

#     def load_ogb(self, dataset, verbose=1):
#         inv_entity_vocab = {}
#         inv_relation_vocab = {}

#         zip_files = glob.glob(os.path.join(dataset.root, "mapping/*.csv.gz"))
#         for zip_file in zip_files:
#             csv_file = utils.extract(zip_file)
#             with open(csv_file, "r") as fin:
#                 reader = csv.reader(fin)
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

#         edge_split = dataset.get_edge_split()
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
#             if "head_neg" in split_dict:
#                 neg_h = torch.as_tensor(split_dict["head_neg"])
#                 neg_t = torch.as_tensor(split_dict["tail_neg"])
#                 negative_heads.append(neg_h)
#                 negative_tails.append(neg_t)
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

#     def split(self, test_negative=True):
#         offset = 0
#         neg_offset = 0
#         splits = []
#         for num_sample, num_sample_with_neg in zip(self.num_samples, self.num_samples_with_neg):
#             if test_negative and num_sample_with_neg:
#                 triplets = self[offset: offset + num_sample]
#                 negative_heads = self.negative_heads[neg_offset: neg_offset + num_sample_with_neg]
#                 negative_tails = self.negative_tails[neg_offset: neg_offset + num_sample_with_neg]
#                 split = OGBLKGTest(triplets, negative_heads, negative_tails)
#             else:
#                 split = torch_data.Subset(self, range(offset, offset + num_sample))
#             splits.append(split)
#             offset += num_sample
#             neg_offset += num_sample_with_neg
#         return splits


# class OGBLKGTest(torch_data.Dataset):

#     def __init__(self, triplets, negative_heads, negative_tails):
#         self.triplets = triplets
#         self.negative_heads = negative_heads
#         self.negative_tails = negative_tails
#         self.num_negative = negative_heads.shape[-1]

#     def __getitem__(self, index):
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
