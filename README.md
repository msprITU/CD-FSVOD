# CD-FSVOD: Cross-Domain Few-Shot Video Object Detection

This repository accompanies the M.Sc. thesis **"Cross-Domain Few-Shot Video Object Detection"** (Istanbul Technical University, 2026). It contains the setup scripts and experiment drivers needed to reproduce the runs reported in the thesis, organized in the same order as the thesis chapters.

The thesis builds on [BHRL](https://github.com/hero-y/BHRL) (CVPR 2022) and extends our EUSIPCO 2025 paper [_Cross-Domain One-Shot Video Object Detection_](https://cmsworkshops.com/EUSIPCO2025/view_paper.php?PaperNum=1825), whose original implementation is available at [CD-BHRL-OTU](https://github.com/msprITU/CD-BHRL-OTU). This repository extends that work with:

- Base training of BHRL on the FSVOD-500 video dataset (within-domain base model)
- 1-shot and M-way k-shot cross-domain fine-tuning on the GOT-10K, LaSOT, and TAO novel-class splits
- Online Target Update (OTU) on long-term video, including sequences where the target repeatedly leaves and re-enters the view

## Repository Layout

```
setup/        environment, code, model, and dataset installation
patches/      corrected helper scripts applied over the BHRL code base
data/         FSVOD-500 base-training annotation and evaluation annotation archives
tools/        dataloader configuration utility
experiments/  one directory per experiment group, in thesis order
```

## Requirements

- A CUDA-enabled Linux machine or cloud GPU instance (Ubuntu 22.04 LTS recommended), running as root
- At least 50 GB of free disk space
- `ipython`  (`pip install ipython`)

## Installation

Clone the repository and move its contents next to `/root`:

```bash
cd /root
git clone https://github.com/hanoglu/CD-FSVOD.git
cd CD-FSVOD
```

**Step 1 — run the 0_setup.sh to download** the modified BHRL code base, the pretrained base models, and the benchmark annotations. sh automatically rewrites the internal paths of the downloaded files according to the `/root/BHRL` layout, and copy the patches under the  `patches/` accordingly:

```bash
cd /root
chmod +x CD-FSVOD/setup/0_setup.sh
./CD-FSVOD/setup/0_setup.sh
```

**Step 2 — install dependencies** (PyTorch, mmcv, mmdet, and the modified parallel-execution files):

```bash
ipython CD-FSVOD/setup/1_install_dependencies.ipy
```

**Step 3 — download the test video sequences** downloads the VOT-long-term sequences specified by vot_videos.txt (edit `BHRL/VOTIMAGES/vot_videos.txt`):

```bash
ipython CD-FSVOD/setup/2_download_vot_sequences.ipy
```
**run the following code line to run the base training on FSVOD-500 dataset, you also need to download FSVOD-500 training data (GOT-10K, LaSOT, and TAO partitions), as well as the test data  required for the few-shot fine-tuning experiments:**

```bash
chmod +x CD-FSVOD/setup/3_download_fsvod500.sh
./CD-FSVOD/setup/3_download_fsvod500.sh
```

**Dataloader configuration.** The evaluation scripts configure the dataloader class list automatically. To set it manually (for example before a standalone run), use:

```bash
ipython CD-FSVOD/tools/configure_dataloader.ipy
```

## Test Cases

The test cases described  below follow the order of the thesis outline: the network is first base-trained, its unadapted cross-domain performance is measured, and the two inference-time adaptation mechanisms (few-shot fine-tuning and Online Target Update) are then evaluated on still-image and long-term video benchmarks.

### 1. Base Training on FSVOD-500
Trains BHRL from scratch on the 320 base classes of the FSVOD-500 training split (batch size 16, SGD with learning rate 0.02, step decay  XX, 9 epochs), producing the within-domain video base model. Note that the annotations of training data  copied from (`data/fsvod_train.json`) to `/root/BHRL` by the setup script.

```bash
./experiments/01_base_training/make_base_config.sh   # derive the training config
./experiments/01_base_training/train_base.sh         # start training
```

The final checkpoint is written to `/root/BHRL/work_dirs/fsvod_base/`. To use it in the test cases described below, copy it to `/root/BHRL/checkpoints/`.

### 2. Direct Evaluation Without Adaptation

Evaluates a base model on novel classes as-is, without any fine-tuning. This isolates the contribution of the base training domain (within-domain FSVOD-500 base vs. cross-domain still-image base) before any adaptation is applied.

```bash
ipython experiments/02_direct_evaluation/eval_no_finetuning.ipy
```

Edit the `item_` (dataset), `CLASS_` (class names), and `split` variables at the top of the script to select the benchmark, the classes, and the base model.

### 3. Few-Shot Cross-Domain Fine-Tuning

Adapts a base model to each novel class at inference time by fine-tuning all network parameters on the support examples (100 iterations), then evaluates on the class test split. The k-shot script sweeps k = 1..5 support examples per class, covering both the 1-shot and the M-way 5-shot protocols on the FSVOD-500 novel classes originating from GOT-10K, LaSOT, and TAO. The required class-based annotations ship with the repository and are installed by the setup script.

```bash
ipython experiments/03_few_shot_finetuning/finetune_kshot.ipy        # k-shot (k = 1..5) fine-tuning per class
ipython experiments/03_few_shot_finetuning/finetune_kshot_video.ipy  # fine-tuning with per-video support sets
```

Each script exposes `ITEMS` (class names), `split` (base model), `DB` (source benchmark: `GOT`, `LaSOT`, or `TAO`), and `RUN_CODE` (output tag) variables at the top. Results and logs are collected under `/root/BHRL/fsvod_results/`.

### 4. Class-Based Still-Image Benchmarks

Evaluates the detector under the class-based protocol on the cross-domain still-image benchmarks (ArTaxOr, DIOR, UODD), with and without fine-tuning. Benchmark images must be downloaded separately as described in [CD-BHRL-OTU](https://github.com/msprITU/CD-BHRL-OTU#dataset-preparation).

```bash
ipython experiments/04_still_image_benchmarks/class_based.ipy            # without fine-tuning
ipython experiments/04_still_image_benchmarks/class_based_finetuned.ipy  # with fine-tuning
```

Edit `item_` and `CLASS_` at the top of each script to select the dataset and its class list.

### 5. Long-Term Video: Fine-Tuning and Online Target Update

Reproduces the long-term video experiments on VOT-LT sequences. First-frame fine-tuning adapts the base model to the target of each sequence; the Online Target Update (OTU) then refreshes the query during the sequence from the model's own high-confidence detections, using the temporal aggregation criteria described in the thesis (score threshold 0.85, IoU threshold 0.7, segments of 100 frames).

First-frame fine-tuning followed by evaluation:

```bash
ipython experiments/05_long_term_video/finetune_coco.ipy [seq ...]  # COCO base model
ipython experiments/05_long_term_video/finetune_voc.ipy  [seq ...]  # VOC base model
```

Fine-tuning combined with the Online Target Update:

```bash
./experiments/05_long_term_video/target_update_coco.sh [seq ...]  # COCO base model
./experiments/05_long_term_video/target_update_voc.sh  [seq ...]  # VOC base model
```

Sequences where the target leaves and re-enters the view (object-absent frames), evaluated with the annotations that keep the absent frames in the timeline:

```bash
./experiments/05_long_term_video/target_update_absent.sh [seq ...]
```

All five scripts run their built-in sequence list by default; passing sequence names as arguments runs only those sequences (e.g. `./target_update_coco.sh cat1 deer`). Results are collected under `/root/BHRL/vot_results/`.

## Pretrained Models

| Model | Base training data | Used by |
|-------|--------------------|---------|
| `model_split1.pth` – `model_split4.pth` | MS COCO (BHRL splits 1–4) | experiments 2–5 |
| `model_voc.pth` | Pascal VOC | experiments 2–5 |
| `model_fsvod.pth` | FSVOD-500 (this thesis) | experiments 2–3 |

All checkpoints are downloaded automatically by the setup script; `model_fsvod.pth` is fetched from this repository's release assets and can also be reproduced from scratch with experiment 1.

## Citation

If you use this repository, please cite the thesis and the EUSIPCO 2025 paper:

```
Hanoglu, Y.K. (2026). Cross-Domain Few-Shot Video Object Detection,
M.Sc. thesis, Istanbul Technical University, Graduate School.

Hanoglu, Y.K., Gunsel, B. and Gurkan, F. (2025). Cross-Domain One-Shot
Video Object Detection, Proceedings of the European Signal Processing
Conference (EUSIPCO), pp. 641-645.
```
