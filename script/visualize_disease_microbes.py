import os
import sys
import pprint

import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util
from reasoning.TorchDrug import core
from reasoning.TorchDrug.utils import comm

def load_vocab(dataset): # Load entity and relation vocabularies.
    entity_vocab = getattr(dataset, "entity_vocab", None)
    relation_vocab = getattr(dataset, "relation_vocab", None)
    if entity_vocab is not None and relation_vocab is not None:
        return list(entity_vocab), list(relation_vocab)

    path = dataset.path
    vocabs = []
    for object in ["entity", "relation"]: # Read entity and relation mapping files.
        vocab_file = os.path.join(path, "%s.txt" % object)
        if not os.path.exists(vocab_file):
            vocab_file = os.path.join(path, "mappings", "%s.txt" % object)
        mapping = {}
        with open(vocab_file, "r") as fin:
            for line in fin:
                k, v = line.strip().split("\t")
                mapping[int(v)] = k
        # vocab = [mapping[t] for t in getattr(dataset, "%s_vocab" % object)]
        vocab = [mapping[idx] for idx in range(len(mapping))]
        vocabs.append(vocab)

    return vocabs # Return vocabularies ordered by numeric ID.


def rank(solver, sample, entity_vocab, relation_vocab, t_entities):
    num_relation = len(relation_vocab)
    h_index, t_index, r_index = sample.unbind(-1) # Unpack head, tail, and relation IDs.
    inverse = torch.stack([t_index, h_index, r_index + num_relation], dim=-1) # Build the inverse triple.
    batch = sample.unsqueeze(0) # Add a batch dimension.
    if sample.ndim == 1: # Use a single visualization query for one-dimensional input.
        vis_batch = torch.stack([sample])
        # vis_batch = torch.stack([sample, inverse]) # Visualize both forward and inverse triples.
    else:
        is_t_neg = (h_index == h_index[0]).all() # Detect tail-negative batches.
        vis_batch = sample[:1] if is_t_neg else inverse[:1] # Select the query direction that matches the negative side.
    batch = batch.to(solver.device) 
    vis_batch = vis_batch.to(solver.device)
    solver.model.eval()
    with torch.no_grad():
        # pred, target = solver.model.predict_and_target(batch,t_entities) # Predict scores and targets.
        pred, target = solver.model.vis_predict_and_target(batch,t_entities) # Predict scores and targets.
    
    # Map global entity IDs to positions inside the typed candidate set.
    t_entities_map = {word: idx for idx, word in enumerate(t_entities)}
    
    if isinstance(target, tuple):
        mask, target = target # Split mask and target when the task returns both.
        tmp = t_entities_map[target.squeeze().item()]
        tmp = torch.tensor([[[tmp]]])
      
        # Map the target ID to its typed-candidate index.
        # pos_pred = pred.gather(-1, target.unsqueeze(-1)) # Gather the positive score.
        pos_pred = pred.gather(-1, tmp) # Gather the positive score.
        pred_squeezed = pred.squeeze()  # Flatten scores from [1, 1, num_candidates] to [num_candidates].
        # Pair each candidate entity with its score.
        pairs = list(zip(t_entities, pred_squeezed.tolist()))
        
        
        # Sort candidates by descending score.
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        # print(pairs_sorted)
        # Print the top-ranked candidates.
        
        
        print("######### RANK LIST ########")
        for i, (entity, pred_value) in enumerate(pairs_sorted[:2000]):
            print(f"RANK: {i + 1}, tail entity: {entity_vocab[entity]}, score: {pred_value}")
        
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1
        # rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
        return pairs_sorted
    
def visualize_raw(solver, sample, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    h_index, t_index, r_index = sample.unbind(-1) # Unpack head, tail, and relation IDs.
    inverse = torch.stack([t_index, h_index, r_index + num_relation], dim=-1) # Build the inverse triple.
    batch = sample.unsqueeze(0) # Add a batch dimension.
    if sample.ndim == 1: # Use a single visualization query for one-dimensional input.
        vis_batch = torch.stack([sample, inverse]) # Visualize both forward and inverse triples.
    else:
        is_t_neg = (h_index == h_index[0]).all() # Detect tail-negative batches.
        vis_batch = sample[:1] if is_t_neg else inverse[:1] # Select the query direction that matches the negative side.
    batch = batch.to(solver.device) 
    vis_batch = vis_batch.to(solver.device)

    solver.model.eval()

    with torch.no_grad():
        pred, target = solver.model.predict_and_target(batch) # Predict scores and targets.
    if isinstance(target, tuple):
        mask, target = target # Split mask and target when the task returns both.
        pos_pred = pred.gather(-1, target.unsqueeze(-1)) # Gather the positive score.
        rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
    else:
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1
        
    # print(">>>>>>>>>>>>>>pos_pred>>>>>>>>>>>>>>")
    # print(pos_pred)
    paths, weights, num_steps = solver.model.visualize(vis_batch) # Retrieve reasoning paths.
    batch = batch.tolist()
    rankings = rankings.tolist()
    paths = paths.tolist()
    weights = weights.tolist()
    num_steps = num_steps.tolist()

    logger.warning("")
    for i in range(len(vis_batch)):
        h, t, r = vis_batch[i] # Decode the visualization query.
        h_token = entity_vocab[h]
        t_token = entity_vocab[t]
        r_token = relation_vocab[r % num_relation]
        if r >= num_relation: # Mark inverse relations in path output.
            r_token += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_token, h_token, r_token, rankings[i]))

        for path, weight, num_step in zip(paths[i], weights[i], num_steps[i]): # Iterate over path weights and lengths.
            if weight == float("-inf"): # Stop when no more valid paths are available.
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
    # Store parsed triples.
    data = []
    try:
        # Read triples line by line.
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip surrounding whitespace.
                line = line.strip()
                if line:  # Skip empty rows.
                    # Split each row into head, relation, and tail.
                    columns = line.split('\t')
                    # Keep only valid triples.
                    if len(columns) == 3:
                        data.append(columns)
                    else:
                        print(f"Warning: skipped malformed triple row: {line}")
        print(f"Loaded {len(data)} triples")
        return data
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error while reading triples: {e}")
        return None


def build_train_triple_set(dataset, file_path=None):
    graph = getattr(dataset, "graph", None)
    num_samples = getattr(dataset, "num_samples", None)
    if graph is not None and num_samples:
        num_train = int(num_samples[0])
        print("Building training triple index from dataset cache/graph...")
        train_triple_set = set()
        for h_index, t_index, r_index in graph.edge_list[:num_train].cpu():
            train_triple_set.add((int(h_index), int(r_index), int(t_index)))
        print(f"Training triple index ready: {len(train_triple_set)} unique triples")
        return train_triple_set

    print("Building training triple index from train.txt...")
    train_data = read_train_data(file_path) if file_path else None
    train_triple_set = set()
    entity_to_id = getattr(dataset, "inv_entity_vocab", None) or {}
    relation_to_id = getattr(dataset, "inv_relation_vocab", None) or {}
    if train_data and entity_to_id and relation_to_id:
        for h_token, r_token, t_token in train_data:
            if h_token in entity_to_id and r_token in relation_to_id and t_token in entity_to_id:
                train_triple_set.add((entity_to_id[h_token], relation_to_id[r_token], entity_to_id[t_token]))
    print(f"Training triple index ready: {len(train_triple_set)} unique triples")
    return train_triple_set


def get_relation_name(relation_vocab, relation_id):
    num_relation = len(relation_vocab)
    name = relation_vocab[relation_id % num_relation]
    if relation_id >= num_relation:
        name += "^(-1)"
    return name


def is_known_training_triple(train_triple_set, head_id, relation_id, tail_id, num_relation):
    base_relation_id = relation_id % num_relation
    if relation_id >= num_relation:
        return (tail_id, base_relation_id, head_id) in train_triple_set
    return (head_id, base_relation_id, tail_id) in train_triple_set


if __name__ == "__main__":
    print("[MAPLE] Starting visualization workflow")
    
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    if torch.cuda.is_available() and "engine" in cfg and "gpus" in cfg.engine and cfg.engine.gpus:
        torch.cuda.set_device(int(cfg.engine.gpus[0]))
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    
    # ---------------------------------------------------------
    # 1. Load model and vocabularies.
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
    # 2. Preload training triples for filtering.
    # ---------------------------------------------------------
    file_path = os.path.join(cfg.dataset.path, "train.txt")
    train_triple_set = build_train_triple_set(dataset, file_path)

    # ---------------------------------------------------------
    # 3. Collect target entities by semantic type.
    # ---------------------------------------------------------
    r_entity = 52 + 156
    t_class = "Microbe"
    print(f"Collecting tail candidates with type: {t_class}")
    t_entities = []
    for entity_idx, entity_str in enumerate(entity_vocab):
        if entity_str.startswith(t_class + "//"):
            t_entities.append(entity_idx)
    print(f"Collected {len(t_entities)} target entities")

    # ---------------------------------------------------------
    # 4. Run ranking and path visualization.
    # ---------------------------------------------------------
    solver.model.eval()
    
    
    
    
    for idx, h_entity in enumerate(h_entitis):
        print(f"\nProcessing head entity {idx+1}/{len(h_entitis)}: {entity_vocab[h_entity]} (ID: {h_entity})")
        
        # Keep inference under no_grad to reduce memory use.
        with torch.no_grad():
            # Build the query triple.
            query_h_t_r = torch.tensor([h_entity, 66100, r_entity])
            
            # Rank typed candidate tails.
            pairs_sorted = rank(solver, query_h_t_r, entity_vocab, relation_vocab, t_entities)
            
            display_count = 0
            max_display = 1
            
            if pairs_sorted:
                for rank_idx, (tail_entity_idx, pred_value) in enumerate(pairs_sorted[:300]):
                    head_name = entity_vocab[h_entity]
                    relation_name = get_relation_name(relation_vocab, r_entity)
                    tail_name = entity_vocab[tail_entity_idx]
                    
                    # Skip known training triples to highlight novel predictions.
                    if not is_known_training_triple(
                        train_triple_set,
                        h_entity,
                        r_entity,
                        tail_entity_idx,
                        len(relation_vocab),
                    ):
                        print(f"RANK: {rank_idx + 1}")
                        print(f"Tail entity: {tail_name} (index: {tail_entity_idx})")
                        print(f"Query triple: ({head_name}, {relation_name}, {tail_name})")
                        print(f"Score: {pred_value}\n")
                        
                        print("=================CHANGE_RICHNESS_DcrMi==================")
                        # Build the query used for path visualization.
                        vis_input = torch.tensor([h_entity, tail_entity_idx, r_entity])
                        
                        visualize_raw(
                            solver,
                            vis_input,
                            entity_vocab,
                            relation_vocab
                        )
                        print("--------------------------------------------------------\n")
                        
                        # Release temporary tensors early.
                        del vis_input
                        
                        display_count += 1
                        if display_count >= max_display:
                            break
            
            # Release large per-query objects.
            del pairs_sorted
            del query_h_t_r
            
            # Clear cached GPU memory after each head entity.
            # This may slow execution slightly but prevents memory growth.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Visualization complete.")
