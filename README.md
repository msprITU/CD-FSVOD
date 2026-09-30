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
git clone https://github.com/msprITU/CD-FSVOD.git
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

**Dataloader configuration.** The BHRL dataloader (`/root/BHRL/mmdet/datasets/one_shot_voc.py`) has to know the category names of the annotation file it reads and which annotated instance serves as the query image. Both are hard-coded in that file, so they are rewritten in place whenever the benchmark changes:

| Line in `one_shot_voc.py` | Meaning | Value |
|---------------------------|---------|-------|
| `SED_PARAMETER_CLASSES` | category names of the benchmark | depends on the dataset, see below |
| `SED_PARAMETER_SELF_SPLIT` | indices of the categories to evaluate | all categories of the dataset |
| `SED_PARAMETER_POSITION_MULTIPLIER` | query position `p` (0–4): of the `N` annotated instances of a class, instance `⌊(N−1)/5⌋·p` is used as the query | `0` = first instance (the first frame of a video) |

| Dataset | Categories |
|---------|------------|
| `VOT` | the 20 Pascal VOC category names used by the VOT-LT annotations |
| `FSVOD500` | the 100 FSVOD-500 novel classes |
| `ArTaxOr` | 7 classes |
| `DIOR` | 20 classes |
| `UODD` | 3 classes |

All experiment scripts do this automatically for the data they evaluate, so no manual step is needed when running the test cases below. To set the configuration manually (for example before calling `tools/test.py` directly), pass the dataset and, optionally, the query position:

```bash
ipython CD-FSVOD/tools/configure_dataloader.ipy VOT         # class list of VOT, query position 0
ipython CD-FSVOD/tools/configure_dataloader.ipy FSVOD500 2  # class list of FSVOD-500, query position 2
```

The script prints the three rewritten lines so the active configuration can be checked.

### Evaluation Annotations

All evaluation annotations are in COCO format and are installed under `/root/BHRL` by the setup script.

**VOT-LT** (`/root/BHRL/vot_annotation/`, shipped as `data/vot_annotation.zip`). Step 3 downloads the complete sequences, including the frames where the target is out of view, to `/root/BHRL/VOTIMAGES/<seq>/`. Which frames are evaluated is decided by the annotation file:

| File | Frames |
|------|--------|
| `<seq>/vot_<seq>_test.json` | only the frames where the target is visible; object-absent frames are dropped (all 50 sequences) |
| `<seq>/vot_<seq>_test_all.json` | every frame of the sequence; object-absent frames are kept and marked with `"bbox": [0, 0, 0, 0]` and `"ignore": 1` (all 50 sequences) |
| `<seq>/vot_<seq>_test_absent.json` | the first frame (query) followed by the object-absent frames only (29 sequences) |
| `<seq>/with_absent/<seq>_part_<n>.json` | `vot_<seq>_test_all.json` cut into 100-frame segments, used by the Online Target Update on object-absent sequences (38 sequences) |
| `<seq>/<seq>_part_<n>.json` | `vot_<seq>_test.json` cut into 100-frame segments, regenerated by the Online Target Update scripts |
| `ft/<seq>_first_ft.json` | the first frame of the sequence, used for first-frame fine-tuning |

38 of the 50 sequences contain object-absent frames; the remaining 12 (`bike1`, `car3`, `car6`, `car8`, `car9`, `car16`, `group1`, `kitesurfing`, `person2`, `person4`, `person5`, `person20`) do not, so their `_test.json` and `_test_all.json` cover the same frames.

**FSVOD-500** (`/root/BHRL/fsvod500_annotation_cb/`, shipped as `data/fsvod500_annotation_cb.zip`). `class_separated_<DB>/<class>.json` holds the test frames of one novel class, `ft_<DB>/` and `ft_video_<DB>/` hold the support sets for fine-tuning, with `<DB>` one of `GOT` (82 classes), `LaSOT` (21 classes), `TAO` (18 classes).

## Test Cases

The test cases described  below follow the order of the thesis outline: the network is first base-trained, its unadapted cross-domain performance is measured, and the two inference-time adaptation mechanisms (few-shot fine-tuning and Online Target Update) are then evaluated on still-image and long-term video benchmarks.

The commands below are run from the repository directory (`cd /root/CD-FSVOD`).

### 1. Base Training on FSVOD-500
Trains BHRL from scratch on the 320 base classes of the FSVOD-500 training split (batch size 16, SGD with learning rate 0.02, step decay  XX, 9 epochs), producing the within-domain video base model. Note that the annotations of training data  copied from (`data/fsvod_train.json`) to `/root/BHRL` by the setup script.

```bash
./experiments/01_base_training/make_base_config.sh   # derive the training config
./experiments/01_base_training/train_base.sh         # start training
```

The final checkpoint is written to `/root/BHRL/work_dirs/fsvod_base/`. To use it in the test cases described below, copy it to `/root/BHRL/checkpoints/`.

### 2. Direct Evaluation Without Adaptation

Takes one of the pretrained base models and tests it directly on cross-domain data, with no fine-tuning and no target update: the checkpoint is loaded, the query is cropped from the first annotated instance of the class (or the first frame of the sequence), and the detector is run on all remaining test frames. Comparing the base models on the same data shows how much the base-training domain alone contributes (FSVOD-500 video base vs. MS COCO / Pascal VOC still-image bases), before any adaptation is applied.

```bash
ipython experiments/02_direct_evaluation/eval_no_finetuning.ipy [key=value ...]
```

Without arguments the script evaluates the FSVOD-500 base model on the GOT-10K class `deer`. The run is selected with the following `key=value` arguments (the same defaults can also be edited in the `CONFIGURE HERE` block at the top of the script):

| Argument | Values | Default | Meaning |
|----------|--------|---------|---------|
| `dataset` | `FSVOD500`, `VOT` | `FSVOD500` | test data: FSVOD-500 novel classes or VOT-LT sequences |
| `db` | `GOT`, `LaSOT`, `TAO` | `GOT` | FSVOD-500 partition (only for `dataset=FSVOD500`) |
| `classes` | comma-separated names, or `all` | `deer` | FSVOD-500 class names (the file names in `fsvod500_annotation_cb/class_separated_<db>/`), or VOT-LT sequence names for `dataset=VOT` |
| `model` | `fsvod`, `split1`, `split2`, `split3`, `split4`, `voc` (comma-separated for several) | `fsvod` | base model, i.e. `/root/BHRL/checkpoints/model_<model>.pth` (see [Pretrained Models](#pretrained-models)) |
| `frames` | `present`, `all`, `absent` | `present` | VOT-LT frame set (only for `dataset=VOT`): `present` drops the object-absent frames, `all` keeps the full timeline, `absent` tests only the object-absent frames |
| `runs` | `1`–`5`, comma-separated | `1` | query position; `1` takes the query from the first annotated instance, `2`–`5` from later ones |
| `dry_run` | `1` | off | print the test commands without running them |

Examples:

```bash
# FSVOD-500 base model on every LaSOT novel class
ipython experiments/02_direct_evaluation/eval_no_finetuning.ipy db=LaSOT classes=all

# compare the FSVOD-500, COCO (split 2) and VOC base models on two TAO classes
ipython experiments/02_direct_evaluation/eval_no_finetuning.ipy db=TAO classes=guitar,fish model=fsvod,split2,voc

# VOT-LT sequences, frames where the target is visible
ipython experiments/02_direct_evaluation/eval_no_finetuning.ipy dataset=VOT classes=deer,carchase model=split2

# VOT-LT sequences, object-absent frames only
ipython experiments/02_direct_evaluation/eval_no_finetuning.ipy dataset=VOT classes=deer,carchase model=split2 frames=absent
```

For `dataset=VOT` the annotation file follows from `frames` (see [Evaluation Annotations](#evaluation-annotations)): `present` uses `vot_<seq>_test.json` and `all` uses `vot_<seq>_test_all.json`. For `absent` the script extracts the first frame and the object-absent frames from `vot_<seq>_test_all.json` into `/root/BHRL/vot_results/no_finetuning/annotations/`, which works for all 38 sequences that have object-absent frames; sequences without any are skipped. With `all` and `absent` the query is always taken from the first frame.

Output is written under `/root/BHRL/fsvod_results/` (FSVOD-500) or `/root/BHRL/vot_results/no_finetuning/` (VOT-LT):

- `logs/<dataset>/<run>.out` — the test log, ending with the COCO AP table of the run
- `results/<run>/results_*.txt` — the top-scoring detection of every tested frame as `index,x,y,w,h,score`, where `index` is the position of the frame in the annotation file; `all_candidates_*.txt` lists all detections

On object-absent frames there is no ground-truth box, so every detection listed for them is a false positive; the scores in `results_*.txt` show how confidently the unadapted model fires when the target is out of view. The AP table is therefore not meaningful for `frames=absent`; instead, for `frames=all` and `frames=absent` the script prints and logs a summary line per sequence with the number of object-absent frames, the mean top score on them, and the number of false detections at the score thresholds 0.5 and 0.85 (`SCORE_THRESHOLDS` in the script; 0.85 is the Online Target Update score threshold).

The cross-domain still-image benchmarks (ArTaxOr, DIOR, UODD) are evaluated without fine-tuning by `class_based.ipy` in test case 4.

### 3. Few-Shot Cross-Domain Fine-Tuning

Adapts a base model to each novel class at inference time by fine-tuning all network parameters on the support examples (100 iterations), then evaluates on the class test split. The k-shot script sweeps k = 1..5 support examples per class, covering both the 1-shot and the M-way 5-shot protocols on the FSVOD-500 novel classes originating from GOT-10K, LaSOT, and TAO. The required class-based annotations ship with the repository and are installed by the setup script.

```bash
ipython experiments/03_few_shot_finetuning/finetune_kshot.ipy        # k-shot (k = 1..5) fine-tuning per class
ipython experiments/03_few_shot_finetuning/finetune_kshot_video.ipy  # fine-tuning with per-video support sets
```

Each script exposes `ITEMS` (class names), `split` (base model), `DB` (source benchmark: `GOT`, `LaSOT`, or `TAO`), and `RUN_CODE` (output tag) variables at the top. `ITEMS` defaults to the GOT-10K classes and must list classes of the selected `DB`, i.e. file names in `/root/BHRL/fsvod500_annotation_cb/class_separated_<DB>/`. Results and logs are collected under `/root/BHRL/fsvod_results/`.

### 4. Class-Based Still-Image Benchmarks

Evaluates the COCO base models under the class-based protocol on the cross-domain still-image benchmarks (ArTaxOr, DIOR, UODD), with and without fine-tuning. The annotations are installed by the setup script; the benchmark images must be downloaded separately (links in [CD-BHRL-OTU](https://github.com/msprITU/CD-BHRL-OTU#dataset-preparation)) and placed so that the test images are found at:

| Dataset | Image directory | Classes |
|---------|-----------------|---------|
| ArTaxOr | `/root/BHRL/data/ARTOXOR/test/` | 7 |
| DIOR | `/root/BHRL/data/DIOR/test/` | 20 |
| UODD | `/root/BHRL/data/UODD/test/` | 3 |

```bash
ipython experiments/04_still_image_benchmarks/class_based.ipy           [dataset ...]  # without fine-tuning
ipython experiments/04_still_image_benchmarks/class_based_finetuned.ipy [dataset ...]  # with fine-tuning
```

Without arguments both scripts run ArTaxOr, DIOR, and UODD in turn; passing dataset names runs only those (e.g. `class_based.ipy UODD`). Every class of a dataset is evaluated, using `<dataset>_annotation/class_separated/<class>.json` for testing and `<dataset>_annotation/ft/<class>_<k>_ft.json` for fine-tuning. `VOT` is accepted as a fourth dataset name and runs the class-based VOT-LT protocol (14 object classes, `vot_annotation/JSON_CLASS_BASED_FINAL/`). The `split` (COCO base model splits) and `i_` (query positions 1–5) variables in each script select the base models and the runs. Results are collected under `/root/BHRL/vot_results/`.

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

Sequences where the target leaves and re-enters the view (object-absent frames), evaluated with the `with_absent/` annotations that keep the absent frames in the timeline (see [Evaluation Annotations](#evaluation-annotations)):

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
