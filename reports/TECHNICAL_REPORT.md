# Technical report — complete after running experiments

**Status:** implementation supplied; actual VOC training and benchmark results are not yet measured. Replace all bracketed fields using your saved artifacts. Never present quick-mode numbers as full-dataset results.

## 1. Problem and practical use

The system detects and localizes 20 everyday object classes in still images. The local application helps inspect images and exports structured bounding-box predictions. It does not cover arbitrary unseen object classes, video tracking or segmentation.

## 2. Dataset and preparation

PASCAL VOC 2012 provides bounding boxes and class labels suited to this task. The pipeline decodes images, validates XML and boxes, clips bounds, converts VOC coordinates and removes exact decoded-pixel duplicates across splits. All exclusions/repairs are logged. Valid difficult objects are used for training but ignored during evaluation.

Record actual data:

- Official train images after cleaning: [count]
- Fit/tune/final evaluation image counts: [counts from results_summary.json]
- Rejected images/boxes and repaired boxes: [cleaning_report.json]
- Duplicate images removed: [count]
- Missing/rare classes: [counts and implications]
- Split seed: [seed]; run name: [name]; quick mode: [must be false for final results]

The fit/tune split is random, not multilabel-stratified. Official validation is reserved for final evaluation, with cleaning exclusions disclosed. No hidden test labels are used. Near-duplicates and overlap with pretrained corpora have not been audited.

## 3. Model choice and training

Faster R-CNN with MobileNetV3-Large FPN balances a clear two-stage detection architecture with a relatively lightweight backbone. COCO weights initialize the backbone, RPN and ROI heads. Corresponding COCO classifier and box-regression rows are copied to VOC classes; thus the baseline already recognizes these categories. Fine-tuning adapts this detector to the target distribution.

Training uses SGD with momentum and weight decay, step LR scheduling, gradient clipping, CUDA mixed precision where available, and checkpoint selection/early stopping on tune mAP50. Horizontal flips update image and box coordinates together. Frozen-backbone and full-backbone settings are compared.

- Hardware, PyTorch/Torchvision, device, CPU threads: [actual values]
- Input resolution, batch size, epochs/patience, LR/scheduler: [actual values]
- Training duration and resource limits: [actual measurements]
- Reproducibility: fixed seeds/splits and saved configs; GPU kernels may remain nondeterministic.

## 4. Experiments and results

Paste the actual trial comparison from `experiments.csv` and the pretrained baseline from `baseline_tune.json`. These are all tuning-set metrics. Describe whether the gain justifies extra compute. Full-backbone LR trials isolate learning rate; no-flip versus full_lr_005 isolates augmentation. State confounding factors when comparing other pairs. Report the number of seeds and variance if measured.

[Insert measured experiment table; do not invent scores.]

Winner: [run/config]. Reason: [tune score plus compute/robustness observations].

Final evaluation uses the fixed winning checkpoint on official validation images. Primary metric is VOC continuous AP at IoU > 0.50 with difficult matches ignored and duplicate matches penalized. The secondary IoU-average metric is VOC-style AP50:95, not official COCO mAP. AP excludes classes without non-difficult GT; disclose class coverage.

[Insert final_comparison.csv and per-class AP50.]

## 5. Error analysis

Use `tune_errors.csv`, `final_errors.csv`, PR curves and saved examples. The error categories are heuristic diagnostics at a confidence threshold of 0.5; they are not equivalent to AP error attribution.

- Example 1: [image ID, ground truth, prediction, specific failure and plausible reason]
- Example 2: [same]
- Example 3: [same]
- Weak classes/sizes/occlusion conditions: [evidence]
- Chosen corrective action and whether it was tested: [details]

## 6. Optimization

The same checkpoint is benchmarked at original and reduced inference resolution. Select the lowest median latency satisfying <=0.02 absolute tune mAP50 loss. Selection uses only tune data; compare fixed policies on held-out evaluation afterward.

[Insert optimization_tune.csv and final_comparison.csv.]

Record batch size, warm-ups, repetitions, median/p95 latency, device, and memory when measured. Model latency excludes disk decoding and UI rendering. CPU memory is not measured. Resolution changes do not reduce checkpoint size or parameter count. A slower or inaccurate candidate is an informative unsuccessful optimization attempt; report it honestly.

## 7. Problems, strengths and limitations

Actual problems encountered and how they were resolved: [describe your run; distinguish observed failures from anticipated issues].

Design strengths: modular readable functions; shared cleaning/preprocessing; explicit split boundaries and class mapping; full experiment logs; portable checkpoint; CPU demo.

Limitations: only 20 classes; limited training budget; random rather than stratified splitting; no near-duplicate or pretraining-overlap audit; no exact interrupted-training resume; no multi-seed confidence estimates by default; sensitivity to small/occluded objects requires empirical assessment; no production robustness or calibration study.

## 8. Future work

Prioritize changes based on observed evidence: multilabel stratification, additional seeds, size-aware error analysis, alternative lightweight detector, carefully validated augmentation, score calibration, deployment quantization/export with parity checks, and near-duplicate analysis. Do not claim any of these was implemented unless you add and verify it.

## 9. Reproduction and artifact links

- GitHub repository: [actual URL]
- Commit hash: [actual hash]
- Trained checkpoint URL and SHA256: [actual values, or precise retraining instructions]
- Demo command: `streamlit run app.py`
- Main workflow: `voc_detection.ipynb`, full mode
- Exact evaluated image IDs: `artifacts/<run>/splits.json`
- AI-use disclosure: `AI_USAGE.md`
