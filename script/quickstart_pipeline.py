#!/usr/bin/env python3

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a connected MiHIKG quickstart graph and run the MAPLE two-stage pipeline."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/quickstart_train.yaml",
        help="Quickstart YAML config relative to the repository root.",
    )
    parser.add_argument("--gpu", default=None, help="Physical GPU index exposed to the pipeline.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build and validate the small graph.")
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild the graph from the private full cache.")
    parser.add_argument("--reuse-distmult", action="store_true", help="Reuse an existing quickstart DistMult checkpoint.")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def normalize_vocab(vocab):
    if isinstance(vocab, (list, tuple)):
        return list(vocab)
    if not isinstance(vocab, dict):
        raise TypeError("Unsupported vocabulary type: %s" % type(vocab).__name__)
    if not vocab:
        return []
    first_key = next(iter(vocab))
    if isinstance(first_key, int):
        return [vocab[index] for index in range(len(vocab))]
    ordered = [None] * len(vocab)
    for token, index in vocab.items():
        ordered[int(index)] = token
    if any(token is None for token in ordered):
        raise ValueError("Vocabulary indexes are not contiguous")
    return ordered


def choose_seed(degree, entity_vocab, prefixes):
    candidate_count = min(10000, degree.numel())
    candidates = degree.topk(candidate_count).indices.tolist()
    for prefix in prefixes:
        for entity_id in candidates:
            if entity_vocab[entity_id].startswith(prefix):
                return entity_id
    return candidates[0]


def choose_initial_seed(triplets, degree, entity_vocab, relation_vocab, config):
    seed_relation = config.get("seed_relation")
    if seed_relation:
        if seed_relation not in relation_vocab:
            raise ValueError("Seed relation is not present in the source graph: %s" % seed_relation)
        relation_id = relation_vocab.index(seed_relation)
        relation_edge_indexes = (triplets[:, 2] == relation_id).nonzero().flatten()
        if relation_edge_indexes.numel() == 0:
            raise RuntimeError("Seed relation has no edges: %s" % seed_relation)
        relation_edges = triplets[relation_edge_indexes]
        edge_priority = degree[relation_edges[:, 0]] + degree[relation_edges[:, 1]]
        top_count = min(100, len(relation_edges))
        top_positions = edge_priority.topk(top_count).indices
        chosen_position = top_positions[int(config["seed"]) % top_count]
        chosen_edge = relation_edges[chosen_position].unsqueeze(0)
        seed_nodes = torch.unique(chosen_edge[:, :2])
        if len(seed_nodes) == 2:
            return seed_nodes, chosen_edge
    seed_id = choose_seed(degree, entity_vocab, config.get("seed_prefixes", []))
    return torch.tensor([seed_id]), torch.empty((0, 3), dtype=torch.long)


def grow_connected_nodes(triplets, num_entity, entity_vocab, relation_vocab, config):
    target_nodes = int(config["target_num_nodes"])
    growth_batch_size = int(config.get("growth_batch_size", 100))
    initial_growth_batch_size = int(config.get("initial_growth_batch_size", growth_batch_size))
    heads = triplets[:, 0]
    tails = triplets[:, 1]
    degree = torch.bincount(torch.cat([heads, tails]), minlength=num_entity)
    seed_nodes, seed_edges = choose_initial_seed(
        triplets,
        degree,
        entity_vocab,
        relation_vocab,
        config,
    )

    selected = torch.zeros(num_entity, dtype=torch.bool)
    selected[seed_nodes] = True
    selected_count = len(seed_nodes)
    tree_edges = [seed_edges] if len(seed_edges) else []
    iteration = 0
    while selected_count < target_nodes:
        iteration += 1
        head_selected = selected[heads]
        tail_selected = selected[tails]
        forward_mask = head_selected & ~tail_selected
        backward_mask = tail_selected & ~head_selected
        boundary_nodes = torch.cat([
            tails[forward_mask],
            heads[backward_mask],
        ])
        boundary_edge_indexes = torch.cat([
            forward_mask.nonzero().flatten(),
            backward_mask.nonzero().flatten(),
        ])
        if boundary_nodes.numel() == 0:
            raise RuntimeError("Connected expansion stopped before reaching the target node count")
        connection_count = torch.bincount(boundary_nodes, minlength=num_entity)
        candidate_ids = connection_count.nonzero().flatten()
        candidate_ids = candidate_ids[~selected[candidate_ids]]
        if candidate_ids.numel() == 0:
            raise RuntimeError("No new connected nodes are available")

        current_growth_size = initial_growth_batch_size if iteration == 1 else growth_batch_size
        add_count = min(
            current_growth_size,
            target_nodes - selected_count,
            candidate_ids.numel(),
        )
        degree_scale = int(degree.max().item()) + 1
        priority = connection_count[candidate_ids] * degree_scale + degree[candidate_ids]
        additions = candidate_ids[priority.topk(add_count).indices]
        addition_mask = torch.zeros(num_entity, dtype=torch.bool)
        addition_mask[additions] = True
        parent_mask = addition_mask[boundary_nodes]
        parent_nodes = boundary_nodes[parent_mask]
        parent_edge_indexes = boundary_edge_indexes[parent_mask]
        parent_order = parent_nodes.argsort()
        sorted_parent_nodes = parent_nodes[parent_order]
        first_parent = torch.ones(len(sorted_parent_nodes), dtype=torch.bool)
        first_parent[1:] = sorted_parent_nodes[1:] != sorted_parent_nodes[:-1]
        chosen_parent_edges = parent_edge_indexes[parent_order[first_parent]]
        if len(chosen_parent_edges) != add_count:
            raise RuntimeError("Failed to assign one connecting edge to every new node")
        tree_edges.append(triplets[chosen_parent_edges])
        selected[additions] = True
        selected_count += add_count
        print(
            "[quickstart] connected expansion %d: selected %d/%d nodes"
            % (iteration, selected_count, target_nodes),
            flush=True,
        )

    induced_mask = selected[heads] & selected[tails]
    induced_edges = torch.unique(triplets[induced_mask], dim=0)
    selected_ids = selected.nonzero().flatten()
    return selected_ids, induced_edges, torch.cat(tree_edges), seed_nodes


def remap_induced_edges(selected_ids, induced_edges, num_source_entities):
    entity_remap = torch.full((num_source_entities,), -1, dtype=torch.long)
    entity_remap[selected_ids] = torch.arange(len(selected_ids))
    local_edges = induced_edges.clone()
    local_edges[:, 0] = entity_remap[local_edges[:, 0]]
    local_edges[:, 1] = entity_remap[local_edges[:, 1]]
    return local_edges


def select_connected_edges(local_edges, tree_edges, num_nodes, target_edges, seed):
    if target_edges < num_nodes - 1:
        raise ValueError("target_num_edges must be at least target_num_nodes - 1")
    if len(local_edges) < target_edges:
        raise RuntimeError(
            "The induced graph has only %d edges; requested %d" % (len(local_edges), target_edges)
        )

    if len(tree_edges) != num_nodes - 1:
        raise RuntimeError("Connected expansion did not produce a complete spanning tree")
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.index_add_(0, tree_edges[:, 0], torch.ones(len(tree_edges), dtype=torch.long))
    degrees.index_add_(0, tree_edges[:, 1], torch.ones(len(tree_edges), dtype=torch.long))
    relation_scale = int(local_edges[:, 2].max().item()) + 1
    local_keys = (local_edges[:, 0] * num_nodes + local_edges[:, 1]) * relation_scale + local_edges[:, 2]
    tree_keys = (tree_edges[:, 0] * num_nodes + tree_edges[:, 1]) * relation_scale + tree_edges[:, 2]
    extra_indexes = (~torch.isin(local_keys, tree_keys)).nonzero().flatten()
    extra_count = target_edges - len(tree_edges)
    if len(extra_indexes) < extra_count:
        raise RuntimeError("The induced graph does not contain enough non-tree edges")

    extra_edges = local_edges[extra_indexes]
    generator = torch.Generator().manual_seed(seed)
    random_order = torch.randperm(len(extra_edges), generator=generator).tolist()
    endpoints = extra_edges[:, :2].tolist()
    degree_list = degrees.tolist()
    leaf_count = sum(value < 2 for value in degree_list)
    chosen_positions = []
    chosen_mask = set()
    for position in random_order:
        head_id, tail_id = endpoints[position]
        if degree_list[head_id] < 2 or degree_list[tail_id] < 2:
            chosen_positions.append(position)
            chosen_mask.add(position)
            if degree_list[head_id] < 2:
                leaf_count -= 1
            if degree_list[tail_id] < 2:
                leaf_count -= 1
            degree_list[head_id] += 1
            degree_list[tail_id] += 1
            if leaf_count == 0 or len(chosen_positions) == extra_count:
                break
    if len(chosen_positions) < extra_count:
        for position in random_order:
            if position not in chosen_mask:
                chosen_positions.append(position)
                if len(chosen_positions) == extra_count:
                    break
    chosen_extras = extra_edges[chosen_positions]
    selected_edges = torch.cat([tree_edges, chosen_extras], dim=0)
    degrees.index_add_(0, chosen_extras[:, 0], torch.ones(len(chosen_extras), dtype=torch.long))
    degrees.index_add_(0, chosen_extras[:, 1], torch.ones(len(chosen_extras), dtype=torch.long))
    return selected_edges, len(tree_edges), degrees


def remap_relations(edges, relation_vocab):
    relation_ids = torch.unique(edges[:, 2], sorted=True)
    relation_remap = torch.full((len(relation_vocab),), -1, dtype=torch.long)
    relation_remap[relation_ids] = torch.arange(len(relation_ids))
    remapped = edges.clone()
    remapped[:, 2] = relation_remap[remapped[:, 2]]
    local_vocab = [relation_vocab[index] for index in relation_ids.tolist()]
    return remapped, local_vocab


def split_edges(edges, tree_size, seed):
    num_edges = len(edges)
    num_train = int(num_edges * 0.8)
    num_valid = int(num_edges * 0.1)
    num_test = num_edges - num_train - num_valid
    mandatory_train = set(range(tree_size))
    for relation_id in torch.unique(edges[:, 2]).tolist():
        relation_edge = int((edges[:, 2] == relation_id).nonzero()[0])
        mandatory_train.add(relation_edge)
    if len(mandatory_train) > num_train:
        raise RuntimeError("Training split is too small to cover all nodes and relations")

    randomizer = random.Random(seed)
    remaining = [index for index in range(num_edges) if index not in mandatory_train]
    randomizer.shuffle(remaining)
    train_indexes = list(mandatory_train)
    train_indexes.extend(remaining[:num_train - len(train_indexes)])
    holdout = remaining[num_train - len(mandatory_train):]
    randomizer.shuffle(train_indexes)
    valid_indexes = holdout[:num_valid]
    test_indexes = holdout[num_valid:num_valid + num_test]
    return edges[train_indexes], edges[valid_indexes], edges[test_indexes]


def write_dataset(output_dir, splits, entity_vocab, relation_vocab, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_edges, valid_edges, test_edges = splits
    for split_name, split_edges_tensor in zip(
        ["train", "valid", "test"],
        [train_edges, valid_edges, test_edges],
    ):
        split_path = output_dir / (split_name + ".txt")
        with split_path.open("w", encoding="utf-8") as stream:
            for head_id, tail_id, relation_id in split_edges_tensor.tolist():
                stream.write(
                    "%s\t%s\t%s\n"
                    % (entity_vocab[head_id], relation_vocab[relation_id], entity_vocab[tail_id])
                )

    mapping_dir = output_dir / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    for name, vocab in [("entity", entity_vocab), ("relation", relation_vocab)]:
        with (mapping_dir / (name + ".txt")).open("w", encoding="utf-8") as stream:
            for index, token in enumerate(vocab):
                stream.write("%s\t%d\n" % (token, index))

    triplets = torch.cat([train_edges, valid_edges, test_edges], dim=0)
    torch.save(
        {
            "triplets": triplets,
            "num_samples": [len(train_edges), len(valid_edges), len(test_edges)],
            "entity_vocab": entity_vocab,
            "relation_vocab": relation_vocab,
        },
        output_dir / "microbekg_transductive_cache.pt",
    )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_dataset(cache_path, expected_nodes=None, expected_edges=None):
    state = torch.load(cache_path, map_location="cpu")
    triplets = state["triplets"]
    num_nodes = len(state["entity_vocab"])
    num_edges = len(triplets)
    if expected_nodes is not None and num_nodes != expected_nodes:
        raise RuntimeError("Quickstart cache has %d nodes; expected %d" % (num_nodes, expected_nodes))
    if expected_edges is not None and num_edges != expected_edges:
        raise RuntimeError("Quickstart cache has %d edges; expected %d" % (num_edges, expected_edges))
    if sum(state["num_samples"]) != num_edges:
        raise RuntimeError("Split sizes do not match the edge count")

    adjacency = [[] for _ in range(num_nodes)]
    degrees = [0] * num_nodes
    for head_id, tail_id, _ in triplets.tolist():
        adjacency[head_id].append(tail_id)
        adjacency[tail_id].append(head_id)
        degrees[head_id] += 1
        degrees[tail_id] += 1
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    if len(visited) != num_nodes:
        raise RuntimeError("Quickstart graph is disconnected")
    if min(degrees) == 0:
        raise RuntimeError("Quickstart graph contains isolated nodes")

    train_size = int(state["num_samples"][0])
    train_edges = triplets[:train_size]
    train_entities = torch.unique(train_edges[:, :2])
    train_relations = torch.unique(train_edges[:, 2])
    if len(train_entities) != num_nodes:
        raise RuntimeError("Not every entity is represented in the training split")
    if len(train_relations) != len(state["relation_vocab"]):
        raise RuntimeError("Not every relation is represented in the training split")
    print(
        "[quickstart] dataset ready: nodes=%d edges=%d relations=%d splits=%s min_degree=%d avg_degree=%.2f"
        % (
            num_nodes,
            num_edges,
            len(state["relation_vocab"]),
            list(state["num_samples"]),
            min(degrees),
            sum(degrees) / num_nodes,
        ),
        flush=True,
    )


def prepare_dataset(config, rebuild):
    quickstart = config["quickstart"]
    output_dir = resolve_path(config["dataset"]["path"])
    cache_path = output_dir / "microbekg_transductive_cache.pt"
    expected_nodes = int(quickstart["target_num_nodes"])
    expected_edges = int(quickstart["target_num_edges"])
    if cache_path.exists() and not rebuild:
        validate_dataset(cache_path, expected_nodes, expected_edges)
        return

    source_cache = resolve_path(quickstart["source_cache"])
    if not source_cache.exists():
        raise FileNotFoundError(
            "The packaged quickstart dataset is missing and the private source cache is unavailable: %s"
            % source_cache
        )
    print("[quickstart] loading private source cache: %s" % source_cache, flush=True)
    source_state = torch.load(source_cache, map_location="cpu")
    source_triplets = source_state["triplets"].long().cpu()
    source_entity_vocab = normalize_vocab(source_state["entity_vocab"])
    source_relation_vocab = normalize_vocab(source_state["relation_vocab"])
    selected_ids, induced_edges, source_tree_edges, seed_nodes = grow_connected_nodes(
        source_triplets,
        len(source_entity_vocab),
        source_entity_vocab,
        source_relation_vocab,
        quickstart,
    )
    local_edges = remap_induced_edges(selected_ids, induced_edges, len(source_entity_vocab))
    local_tree_edges = remap_induced_edges(selected_ids, source_tree_edges, len(source_entity_vocab))
    print(
        "[quickstart] induced graph: nodes=%d unique_edges=%d"
        % (len(selected_ids), len(local_edges)),
        flush=True,
    )
    selected_edges, tree_size, degrees = select_connected_edges(
        local_edges,
        local_tree_edges,
        len(selected_ids),
        expected_edges,
        int(quickstart["seed"]),
    )
    selected_edges, local_relation_vocab = remap_relations(selected_edges, source_relation_vocab)
    splits = split_edges(selected_edges, tree_size, int(quickstart["seed"]))
    local_entity_vocab = [source_entity_vocab[index] for index in selected_ids.tolist()]
    manifest = {
        "description": "Connected MiHIKG quickstart subset for the MAPLE two-stage training pipeline",
        "seed": int(quickstart["seed"]),
        "source_seed_entities": [source_entity_vocab[index] for index in seed_nodes.tolist()],
        "source_seed_relation": quickstart.get("seed_relation"),
        "num_nodes": len(local_entity_vocab),
        "num_edges": len(selected_edges),
        "num_relations": len(local_relation_vocab),
        "split_sizes": [len(split) for split in splits],
        "split_ratio": [0.8, 0.1, 0.1],
        "connected": True,
        "minimum_degree_before_split": int(degrees.min().item()),
        "full_private_graph_required_at_runtime": False,
    }
    write_dataset(output_dir, splits, local_entity_vocab, local_relation_vocab, manifest)
    validate_dataset(cache_path, expected_nodes, expected_edges)


def run_pipeline(config_path, config, gpu, reuse_distmult):
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    distmult_checkpoint = resolve_path(config["pretrain_gen_model"])

    if not reuse_distmult:
        if distmult_checkpoint.exists():
            distmult_checkpoint.unlink()
        print("[quickstart] stage 1/2: pretraining DistMult", flush=True)
        subprocess.run(
            [sys.executable, "pretrain.py", "-c", str(config_path)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    if not distmult_checkpoint.exists():
        raise RuntimeError("DistMult checkpoint was not created: %s" % distmult_checkpoint)

    print("[quickstart] stage 2/2: MAPLE reinforcement training with frozen DistMult backbone", flush=True)
    subprocess.run(
        [sys.executable, "script/train.py", "-c", str(config_path)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    print("[quickstart] pipeline completed", flush=True)


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    prepare_dataset(config, args.rebuild_data)
    if not args.prepare_only:
        run_pipeline(config_path, config, args.gpu, args.reuse_distmult)


if __name__ == "__main__":
    main()
