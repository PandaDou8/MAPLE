import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from reasoning.TorchDrug import core
from reasoning.TorchDrug.utils import comm
from reasoning import dataset, layer, model, task, util


QUERY_ENTITIES = [
    "Disease//MONDO:0002104",
    "Disease//MONDO:0005146",
    "Disease//MONDO:0007739",
]
QUERY_RELATION = "CHANGE_RICHNESS_DcrMi"
TARGET_PREFIX = "Microbe//"


def load_vocab(dataset):
    return list(dataset.entity_vocab), list(dataset.relation_vocab)


def load_known_training_triples(dataset, head_ids, relation_id):
    train_path = Path(dataset.path) / "train.txt"
    if train_path.exists():
        head_names = {dataset.entity_vocab[idx] for idx in head_ids}
        relation_name = dataset.relation_vocab[relation_id]
        known = set()
        with train_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                head, relation, tail = line.rstrip("\n").split("\t")
                if head in head_names and relation == relation_name:
                    known.add((head, relation, tail))
        return known

    train_size = int(dataset.num_samples[0])
    train_edges = dataset.graph.edge_list[:train_size]
    head_set = set(head_ids)
    known = set()
    for head, tail, relation in train_edges.tolist():
        if head in head_set and relation == relation_id:
            known.add((dataset.entity_vocab[head], dataset.relation_vocab[relation], dataset.entity_vocab[tail]))
    return known


def format_path(path, num_steps, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    triplets = []
    for head, tail, relation in path[:num_steps]:
        relation_name = relation_vocab[relation % num_relation]
        if relation >= num_relation:
            relation_name += "^(-1)"
        triplets.append({
            "head": entity_vocab[head],
            "relation": relation_name,
            "tail": entity_vocab[tail],
        })
    return triplets


def predict_typed_candidates(solver, head_id, relation_id, target_id, candidate_ids, entity_vocab, relation_vocab, known, topk):
    query = torch.tensor([[head_id, target_id, relation_id]], device=solver.device)
    solver.model.eval()
    with torch.no_grad():
        pred, target = solver.model.predict_and_target(query)

    scores = pred[0, 0]
    target_score = float(scores[target_id])
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    candidate_scores = scores[candidate_tensor]
    order = torch.argsort(candidate_scores, descending=True)
    ranked_ids = candidate_tensor[order].tolist()

    ranked = []
    selected = None
    head_name = entity_vocab[head_id]
    relation_name = relation_vocab[relation_id]
    for rank, tail_id in enumerate(ranked_ids, start=1):
        tail_name = entity_vocab[tail_id]
        item = {
            "rank": rank,
            "tail_id": int(tail_id),
            "tail": tail_name,
            "score": float(scores[tail_id]),
            "known_training_triple": (head_name, relation_name, tail_name) in known,
        }
        if rank <= topk:
            ranked.append(item)
        if selected is None and rank <= topk and not item["known_training_triple"]:
            selected = item

    mask = target[0][0, 0]
    full_filtered_rank = int(((target_score <= scores) & mask).sum().item() + 1)
    typed_raw_rank = ranked_ids.index(target_id) + 1
    return {
        "query": {
            "head_id": int(head_id),
            "head": head_name,
            "relation_id": int(relation_id),
            "relation": relation_name,
        },
        "target": {
            "tail_id": int(target_id),
            "tail": entity_vocab[target_id],
            "score": target_score,
        },
        "typed_raw_rank": typed_raw_rank,
        "full_filtered_rank": full_filtered_rank,
        "top_typed_candidates": ranked,
        "selected_unseen_candidate": selected,
        "scores": scores,
    }


def visualize_paths(solver, result, entity_vocab, relation_vocab):
    selected = result["selected_unseen_candidate"]
    if selected is None:
        return []
    query = result["query"]
    vis_batch = torch.tensor(
        [[query["head_id"], selected["tail_id"], query["relation_id"]]],
        dtype=torch.long,
        device=solver.device,
    )
    paths, weights, num_steps = solver.model.visualize(vis_batch)
    formatted = []
    for path, weight, step_count in zip(paths[0].tolist(), weights[0].tolist(), num_steps[0].tolist()):
        if weight == float("-inf"):
            break
        formatted.append({
            "weight": float(weight),
            "steps": format_path(path, int(step_count), entity_vocab, relation_vocab),
        })
    return formatted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--output", default="reproduction.json")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--gpu", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = util.load_config(args.config)
    if args.gpu is not None:
        cfg.engine.gpus = [args.gpu]
    if torch.cuda.is_available() and cfg.engine.gpus:
        torch.cuda.set_device(int(cfg.engine.gpus[0]))
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + comm.get_rank())

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    entity_vocab, relation_vocab = load_vocab(dataset)
    entity_to_id = {name: idx for idx, name in enumerate(entity_vocab)}
    relation_to_id = {name: idx for idx, name in enumerate(relation_vocab)}
    relation_id = relation_to_id[QUERY_RELATION]
    head_ids = [entity_to_id[name] for name in QUERY_ENTITIES]
    target_ids = [
        entity_to_id["Microbe//Parabacteroides johnsonii DSM 18315"],
        entity_to_id["Microbe//Bacteroidetes"],
        entity_to_id["Microbe//Prevotella copri"],
    ]
    microbe_ids = [idx for idx, name in enumerate(entity_vocab) if name.startswith(TARGET_PREFIX)]
    known = load_known_training_triples(dataset, head_ids, relation_id)

    results = []
    for head_id, target_id in zip(head_ids, target_ids):
        result = predict_typed_candidates(
            solver,
            head_id,
            relation_id,
            target_id,
            microbe_ids,
            entity_vocab,
            relation_vocab,
            known,
            args.topk,
        )
        result["candidate_count"] = len(microbe_ids)
        result["paths"] = visualize_paths(solver, result, entity_vocab, relation_vocab)
        result.pop("scores")
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(working_dir) / output_path
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved reproduction results to %s" % output_path)


if __name__ == "__main__":
    main()
