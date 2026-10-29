
# Object Detection on PASCAL VOC 2012

A readable notebook-first project implementing data cleaning, preprocessing, pretrained detection, fine-tuning, controlled experiments, evaluation, error analysis, inference optimization and a local image-upload demo.

**Start with `voc_detection.ipynb`.** The notebook has 11 numbered sections, with short reusable functions in `src/`. It is intentionally unexecuted: no scores, trained weights or dataset are supplied. Training produces the evidence needed for submission.

## Run locally

Use Python 3.11 or 3.12. Open a terminal inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# CPU reference installation; suitable for tests and the local demo:
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
# For an NVIDIA GPU, use the compatible CUDA wheel command instead, e.g. CUDA 12.6:
# python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
jupyter lab voc_detection.ipynb
```

Use the official [installation matrix](https://pytorch.org/get-started/previous-versions/) for your driver/platform. Do not mix arbitrary torch/vision versions. The versions above are a reference pair, not a claim about the newest release. Dependency ranges other than torch/vision are not an exact environment lock; the notebook records actual versions of core model libraries and exports run configuration.

Run cells in order. Leave `QUICK_RUN=True` for the first plumbing check. Then set `QUICK_RUN=False`, choose a fresh run directory and run again for actual submission results. Full mode runs four experiments, up to six epochs each, with early stopping. Runtime depends on hardware and session limits; no guaranteed completion time. If GPU memory is tight, set batch_size=1 or decrease resolution. Increasing resolution often helps small objects but costs compute; verify empirically.

Downloading VOC needs internet access and several GB of disk space. Reserve roughly 8–10 GB including the compressed archive. `torchvision.datasets.VOCDetection` handles downloading and archive verification. If the upstream host is unavailable, download the official VOC2012 train/val archive using the [VOC site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), extract locally and set `VOC_DIR` to its `VOCdevkit/VOC2012` directory. Do not use hidden test annotations or merge official val into training.

## Free Colab / Kaggle workflow

1. Enable an available free GPU runtime. Upload and extract `voc_detection_project.zip`.
2. Change directory into `voc_detection` **before** executing notebook cells. On Colab:

   ```python
   from google.colab import files
   files.upload()  # select voc_detection_project.zip
   import zipfile
   with zipfile.ZipFile('voc_detection_project.zip') as z:
       z.extractall('/content')
   %cd /content/voc_detection
   ```

3. Open the included notebook in Colab (File → Upload notebook). The extracted folder must still be present in `/content`.
4. Inspect the preinstalled torch/vision versions and CUDA availability. Keep a compatible working CUDA pair. Install other missing dependencies, restart the kernel if prompted, and rerun from setup. The pinned pair is the locally smoke-tested reference; other pairs need their own verification.
5. Run quick mode, then full mode. Enable internet in Kaggle for downloads. Free runtimes are ephemeral; save completed trial folders and trained artifacts externally before disconnecting. The notebook includes a ZIP download cell. You can instead set `RUN_DIR` to mounted persistent storage.

All experiments restart from pretrained weights. Inference checkpoints do not include optimizer/scaler/RNG state, so an interrupted experiment restarts from the beginning. Completed checkpoints can be evaluated manually with `load_checkpoint`; do not rerun a trial into a nonempty directory.

## Demo after training

The notebook exports `artifacts/best.pt` and `artifacts/deployment.json`.

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a new image, adjust the confidence threshold, see labeled boxes and download JSON predictions. Download the trained artifacts from Colab and place both files in this project's local `artifacts/` folder to run the app on CPU. No paid cloud deployment is needed. Pretrained weights are not downloaded when loading an exported checkpoint.

## Design and evaluation

- **Model:** Faster R-CNN with MobileNetV3-Large FPN. COCO pretrained detector with explicit VOC class-row transfer, then VOC fine-tuning.
- **Data:** official `train` → deterministic 80/20 fit/tune split. Official `val` → held-out final evaluation. Global exact decoded-pixel duplicates are removed, preserving validation first. No near-duplicate detection or pretraining-corpus overlap audit.
- **Cleaning:** decode images; parse XML; use actual image dimensions; convert 1-based inclusive coordinates; clip bounds; remove invalid boxes; log all changes. Background images are supported.
- **Augmentation:** horizontal flip with box updates; no augmentation in evaluation. The model performs resize/normalization internally.
- **Difficult instances:** included as valid supervision during training. During evaluation, a detection whose best same-class match is difficult contributes neither TP nor FP; difficult instances do not count toward recall.
- **Metrics:** continuous VOC AP at IoU > 0.50, class AP, precision–recall curves and a custom VOC-style average over IoU 0.50:0.05:0.95. The latter is NOT official COCO mAP. Classes absent from a split are explicitly undefined and excluded from means. Evaluation uses confidence floor 0.001 and max 100 detections/image.
- **Experiments:** frozen backbone; full backbone at LR 0.005; full backbone at LR 0.001; no-flip ablation at LR 0.005. Selection uses tune mAP50. Full-backbone LR trials and the augmentation ablation isolate a single factor. The frozen versus full comparison does not isolate LR if compared against the 0.001 trial.
- **Optimization:** same checkpoint, smaller inference resolution. Choose median latency subject to <=0.02 absolute tuning mAP50 drop. Report both policies on final evaluation after freezing selection. No promise that optimization improves latency. Resolution does not shrink the checkpoint.
- **Integrity:** baseline and trial tuning results must not be compared with final held-out scores as if they used the same split. Full validation is an evaluation set, not an official VOC test result. Pretraining on overlapping classes is disclosed; image overlap is unverified.

## Files and generated outputs

```text
voc_detection.ipynb          Main end-to-end workflow
src/config.py               Editable configuration and reproducibility
src/data.py                 Download, cleaning, split and preprocessing
src/model.py                Model adaptation and portable checkpoints
src/engine.py               Training, prediction and benchmarking
src/metrics.py              VOC AP and diagnostic errors
src/visualize.py             Box drawing
app.py                      CPU Streamlit demo
requirements.txt            Dependencies
reports/TECHNICAL_REPORT.md  Report guide to complete with actual results
reports/EXPERIMENTS.md       Planned comparisons and interpretation
reports/DEFENSE_GUIDE.md     Code-understanding and defense questions
AI_USAGE.md                 AI disclosure draft to review
VALIDATION.md               Checks performed on the delivered source
artifacts/<run>/             Generated metrics, plots, checkpoints, audit/splits
```

`artifacts/<run>/experiments.csv`, `optimization_tune.csv`, `final_comparison.csv`, `results_summary.json`, trial histories, per-class metrics, errors, saved prediction examples and split manifests are produced by execution. `clean_manifest.json` contains absolute image paths for that machine; rerun the cleaning cell after moving the data. `splits.json` also records stable image IDs.

## Validation and submission

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic data, no downloads. They cover box conversion/clipping, empty targets, flips, duplicate cleaning, split separation, AP/ignored difficult cases, a real detector forward/backward/checkpoint roundtrip and a one-epoch training-loop integration check. These checks do not replace full training or a browser check of the demo.

Create a GitHub repository, upload source (exclude dataset and large artifacts), complete the report with real full-run metrics, and provide trained weights through an appropriate release/download link or the clear retraining instructions above. Do not commit API keys or private images. For a trained artifact, add its actual SHA256 and download link to the report. No GitHub repository is created by this package.

## Sources

- [Torchvision detector](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)
- [Detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [VOC 2012 development kit](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)
- [PyTorch installation pairs](https://pytorch.org/get-started/previous-versions/)

VOC images retain their source rights; consult dataset terms before redistributing images. This package contains no dataset images.
