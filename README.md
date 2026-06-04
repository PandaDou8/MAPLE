# MAPLE

MAPLE (Memory-Augmented Policy Learning for Evidence-aware reasoning) is a knowledge-guided reinforcement learning framework for interpretable microbe-host mechanism discovery on MiHIKG / MicrobeKG.

This repository contains the runnable MAPLE model code, curated configuration files, visualization scripts, mapping files, and example figures used to demonstrate disease-to-microbe reasoning.

## Overview

The manuscript **“Deciphering Microbe–Host Molecular Cascades via Knowledge-Guided Reinforcement Learning”** introduces two connected components:

- **MiHIKG**: a large-scale, multi-source microbe-human interaction knowledge graph integrating microbes, metabolites, chemicals, host genes, immune factors, diseases, and other biomedical entities.
- **MAPLE**: an interpretable reasoning engine that combines A*Net-style path reasoning with a memory-augmented policy sampler for biologically plausible hard negatives.

MAPLE is designed to move beyond shallow co-occurrence prediction by ranking candidate microbe-host associations and exposing multi-hop evidence paths that connect diseases, microbes, metabolites, host targets, and regulatory mechanisms.

## Figures

### MiHIKG Knowledge Graph

![Overview of the MiHIKG knowledge graph](imgs/fig1.svg)

Figure 1 summarizes the MiHIKG semantic infrastructure, including body-site microbial communities, disease associations, and the multi-scale ontology connecting microbes, metabolites, host genes, diseases, and environmental factors.

### Topological and Functional Landscape

![Topological architecture and metabolic functional landscape of MiHIKG](imgs/fig2.svg)

Figure 2 highlights the metabolite / chemical-centered topology of MiHIKG, showing how small molecules act as a bridge between microbial and host subnetworks.

### MAPLE Framework

![Structural logic and translational application scenarios of MAPLE](imgs/fig3.png)

Figure 3 illustrates the MAPLE workflow: strict negative candidate construction, prior-policy-memory hard negative scoring, A*Net evidence-aware path reasoning, reward feedback, and downstream translational use cases.

## Repository Layout

```text
MAPLE_main02/
├── base_model.py                 # Shared embedding-model utilities
├── memory_distmult.py            # MAPLE memory-augmented DistMult generator
├── pretrain.py                   # DistMult generator pretraining entry
├── config.py                     # Legacy config loader used by training code
├── configs/
│   ├── 对抗6层-logger.yaml        # Full adversarial MAPLE training config
│   ├── reasoning.yaml            # Reasoning / visualization config template
│   └── quickstart_visualization.yaml
├── checkpoints/
│   └── maple_checkpoint.pth      # Local MAPLE checkpoint used by quickstart
├── data/
│   ├── mappings/                 # Small entity / relation mapping tables
│   └── external/                 # Optional place for full MicrobeKG data
├── imgs/                         # README figures
├── reasoning/                    # A*Net / TorchDrug-based KGC engine
├── script/
│   ├── train.py                  # MAPLE training / evaluation entry
│   ├── visualize_disease_microbes.py
│   └── visualize_cad_metabolite.py
└── run.sh                        # Quickstart script
```

## Method Summary

MAPLE trains a path reasoner and a hard-negative sampler together:

1. **Strict candidate generation** builds negative candidates under type and graph constraints.
2. **Frozen relational prior** provides global DistMult-style plausibility scores.
3. **Query-conditioned policy head** adapts negative sampling to the current query context.
4. **Relation-conditioned memory module** stores reward-guided feedback for confusing relation-specific candidates.
5. **A*Net evidence reasoning** expands compact query-relevant subgraphs and scores candidates through multi-hop biomedical paths.
6. **Reward feedback** updates the sampler when hard negatives violate ranking margins, encouraging biologically meaningful discrimination.

## Environment

The code expects a Python environment with PyTorch and graph-learning dependencies available. Typical packages include:

- `torch`
- `numpy`
- `pyyaml`
- `easydict`
- `jinja2`
- `tqdm`
- `torch-scatter`
- `torch-sparse`
- `scikit-learn`
- `matplotlib`

The repository vendors a local `reasoning/TorchDrug/` implementation, so run commands from the repository root.

## Data and Checkpoints

The quickstart config uses the local full MicrobeKG data path:

```text
/dataStor/home/yqzhang/data/mic_data_0523/
```

Expected files:

```text
train.txt
valid.txt
test.txt
entities.dict
relations.dict
entity.txt
relation.txt
microbekg_transductive_cache.pt
```

The quickstart checkpoint is:

```text
checkpoints/maple_checkpoint.pth
```

If you move the dataset or checkpoint, update:

```text
configs/quickstart_visualization.yaml
```

## Quickstart

Run the disease-to-microbe visualization example:

```bash
bash run.sh
```

`run.sh` intentionally keeps the training command commented out and only executes:

```bash
python script/visualize_disease_microbes.py -c configs/quickstart_visualization.yaml
```

The script loads the MAPLE checkpoint, ranks candidate microbes for predefined disease heads, filters known training triples, and prints top novel predictions with interpretable path evidence.

## Training

Full adversarial training is expensive and is disabled in `run.sh` by default. To run it manually, first review GPU, output, dataset, checkpoint, and epoch settings in:

```text
configs/对抗6层-logger.yaml
```

Then run:

```bash
python script/train.py -c configs/对抗6层-logger.yaml
```

## Notes

- `script/visualize_disease_microbes.py` currently uses a fixed CUDA device in the script and a fixed disease-head list for the example case.
- `outputs/`, logs, temporary caches, and large local data files are ignored by Git.
- Large checkpoints and the full MicrobeKG dataset should be distributed separately from source code when preparing a clean release.
