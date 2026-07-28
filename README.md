# OTF-CBM

Official PyTorch implementation of **Bridging Vision and Language Concepts
through Optimal Transport Semantic Flow**.

Accepted to **ECCV 2026 as an Oral Presentation**.
[Paper](https://arxiv.org/abs/2606.26891). If you are attending ECCV, feel
free to reach out for a coffee chat about this work or related research.

## Method

OTF-CBM has two stages:

1. Train a shared inverse-OT cost on PACO using frozen DINOv2 ViT-L/14 image
   features and frozen OpenCLIP ViT-L/14 text features.
2. For each downstream dataset, load the shared cost, freeze its 18
   coefficients `theta`, and optimize the visual projection adapter, CLS
   condition adapter, velocity field, and classifier.

DINOv2 remains frozen. Stage 2 uses class labels only for the auxiliary
flow-matching loss. Evaluation and inference call `model.predict(images)` and
do not use labels.

Supported Stage-2 datasets:

| Dataset | Config | Classes | Reference concepts |
| --- | --- | ---: | ---: |
| CUB-200-2011 | `configs/cub.yaml` | 200 | 312 |
| CIFAR-100 | `configs/cifar100.yaml` | 100 | 892 |
| ImageNet-1K | `configs/imagenet.yaml` | 1000 | 4751 |
| Places365 | `configs/places365.yaml` | 365 | 2544 |
| AwA2 | `configs/awa2.yaml` | 50 | user supplied |

The classifier dimension is inferred from the generated concept bank; these
reference counts are not hard-coded, and custom concept counts are supported.

## Installation

Python 3.9 or later is required. Install the CUDA build of PyTorch appropriate
for your system, then install the repository:

```bash
cd OTF-CBM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The default FAISS backend is `faiss-cpu`. It may be replaced with a compatible
FAISS GPU installation.

Place the pretrained encoders at:

```text
data/pretrained/
├── dinov2_vitl14_pretrain.pth
└── open_clip_pytorch_model.bin
```

DINOv2 code is loaded through PyTorch Hub. For offline use, clone
[DINOv2](https://github.com/facebookresearch/dinov2) and override:

```bash
--set backbone.source=local \
--set backbone.repository=/path/to/dinov2
```

## Data

Use the following layout:

```text
data/
├── datasets/
│   ├── coco/{train2017,val2017}/
│   ├── CUB_200_2011/
│   │   ├── images/
│   │   ├── images.txt
│   │   ├── image_class_labels.txt
│   │   ├── train_test_split.txt
│   │   ├── bounding_boxes.txt
│   │   └── classes.txt
│   ├── cifar100/cifar-100-python/
│   ├── imagenet/{train,val}/<class>/
│   ├── places365/
│   └── Animals_with_Attributes2/JPEGImages/
├── paco/
│   ├── raw/{paco_lvis_v1_train.json,paco_lvis_v1_val.json}
│   ├── paco_image_level_train.json
│   └── paco_image_level_val.json
└── concepts/
    ├── <dataset>_global.txt
    ├── <dataset>_class_concepts.json
    ├── <dataset>_classes.txt
    └── cache/<dataset>.pt
```

Dataset sources:
[PACO](https://github.com/facebookresearch/paco),
[COCO 2017](https://cocodataset.org/dataset/detection-2017.htm),
[CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/),
[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html),
[ImageNet](https://www.image-net.org/download.php),
[Places365](https://github.com/CSAILVision/places365), and
[AwA2](https://cvml.ista.ac.at/AwA2/).

CUB uses the official train/test split and bounding-box crops. The `val`
interface refers to the official CUB test partition. AwA2 uses a deterministic
stratified 80/20 image split over all 50 classes, not its zero-shot split.
ImageNet `train` and `val` must both use ImageFolder-style class directories.

## Concept features required by Stage 2

Before Stage 2, prepare three text files for the selected dataset:

- `<dataset>_global.txt`: one bottleneck concept per line.
- `<dataset>_class_concepts.json`: class name to non-empty description list.
- `<dataset>_classes.txt`: one JSON key per line in dataset label order.

For example, `cub_global.txt` may contain:

```text
has a red wing
has a pointed bill
```

The corresponding class-description JSON uses:

```json
{
  "Black_footed_Albatross": [
    "a large seabird with dark plumage",
    "a bird with a long hooked bill"
  ]
}
```

List the same JSON keys in `cub_classes.txt`, one per line and in CUB label
order.

The reference global vocabularies used for the reported experiments are:

- CUB: the 312-entry
  [`my_cub_concepts.txt`](https://github.com/NMS05/Improving-Concept-Alignment-in-Vision-Language-Concept-Bottleneck-Models/blob/main/CUB/data/my_cub_concepts.txt).
- CIFAR-100, ImageNet, and Places365:
  `<dataset>_filtered.txt` from
  [Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM).

Class-specific descriptions use Label-free-CBM's
`gpt3_init/gpt3_<dataset>_important.json`. AwA2 has no fixed reference
vocabulary in this repository.

The downstream datasets do not require per-image concept annotations. More
fine-grained or domain-specific concepts may be generated with an LLM. A
changed global vocabulary changes the bottleneck dimension and requires a new
Stage-2 classifier.

Run the following command once for each dataset before training Stage 2:

```bash
python scripts/prepare_concepts.py --config configs/cub.yaml
```

This replaces the former `extract_concept_features.py` step. It encodes both
the global bottleneck vocabulary and class-specific descriptions with the
configured OpenCLIP text encoder, then writes
`data/concepts/cache/cub.pt`. Stage-2 training, evaluation, and inference load
this concept bank automatically. Replace `configs/cub.yaml` with the config for
another dataset.

To use an OpenCLIP registered tag instead of a local file:

```bash
python scripts/prepare_concepts.py --config configs/cub.yaml \
  --set text_encoder.pretrained=laion2b_s32b_b82k
```

## Stage 1: shared PACO cost

Convert PACO train and validation annotations:

```bash
python scripts/preprocess_paco.py \
  --annotations data/paco/raw/paco_lvis_v1_train.json \
  --image-root data/datasets/coco \
  --output data/paco/paco_image_level_train.json

python scripts/preprocess_paco.py \
  --annotations data/paco/raw/paco_lvis_v1_val.json \
  --image-root data/datasets/coco \
  --output data/paco/paco_image_level_val.json
```

Train the shared cost once:

```bash
python scripts/train_iot.py --config configs/stage1_paco.yaml
```

This runs 5 theta-only epochs, 5 adapter-only epochs, and 10 joint epochs.
Stage-2 configs load `outputs/stage1_paco/iot_final.pt`; `iot_best.pt` is also
kept using the PACO validation loss. Reduce `training.batch_size` if necessary.

## Stage 2: dataset-specific training

Single GPU:

```bash
python scripts/train.py --config configs/cub.yaml \
  --set training.batch_size=16 \
  --set output.directory=outputs/cub_seed1111
```

Three GPUs with a global batch size of approximately 64:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=3 \
  --master_addr=127.0.0.1 \
  --master_port=29617 \
  scripts/train.py \
  --config configs/cub.yaml \
  --set training.batch_size=21 \
  --set training.num_workers=8 \
  --set output.directory=outputs/cub_ddp_seed1111
```

`training.batch_size` is per GPU under DDP. Replace `configs/cub.yaml` with any
other Stage-2 config. Rank 0 writes:

```text
outputs/<run>/
├── best.pt
├── last.pt
├── epoch_005.pt
├── epoch_010.pt
└── metrics.jsonl
```

Stage-2 checkpoints exclude the frozen DINOv2 state and concept embeddings.
Use the same concept bank and class order for training, evaluation, and
inference.

The three-GPU CUB command above reached **90.06%** (5218/5794) on the official
test split with seed 1111; the best checkpoint was at epoch 18.

## Evaluation and inference

Evaluate a dataset checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/cub.yaml \
  --checkpoint outputs/cub_ddp_seed1111/best.pt \
  --split val
```

Run single-image inference:

```bash
python scripts/infer.py \
  --config configs/cub.yaml \
  --checkpoint outputs/cub_ddp_seed1111/best.pt \
  --image path/to/image.jpg \
  --top-classes 5 \
  --top-concepts 10
```

## Checks

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## Citation

```bibtex
@article{zhang2026bridging,
  title={Bridging Vision and Language Concepts through Optimal Transport Semantic Flow},
  author={Zhang, Chenyang and Dong, Anqi and Zhu, Guangming and Xiong, Nuoye and Wang, Siyuan and Mei, Lin and Zhang, Liang},
  journal={arXiv preprint arXiv:2606.26891},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
