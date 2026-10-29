# Experiment protocol

These are planned experiments, not measured results. Default full mode uses six epochs per trial at most, early stopping patience three, fixed seed 42 and the same fit/tune images. The main config is in the notebook and `src/config.py`.

| Run | Train backbone | LR | Flip probability | Purpose |
|---|---|---|---|---|
| pretrained baseline | no training | n/a | 0 at evaluation | Benefit already provided by COCO weights |
| frozen_backbone | no | 0.005 | 0.5 | Cheaper adaptation of RPN and ROI heads |
| full_lr_005 | yes | 0.005 | 0.5 | Unfreeze compared with same-LR frozen trial |
| full_lr_001 | yes | 0.001 | 0.5 | Controlled LR comparison |
| full_no_flip | yes | 0.005 | 0.0 | Controlled horizontal-flip ablation |

`model.backbone` includes the FPN, so freezing the backbone also freezes FPN parameters. RPN and ROI heads remain trainable. Frozen normalization buffers are fixed in every trial.

All trials start from the same pretrained weights and aligned VOC head. Checkpoint selection uses tune mAP50 per epoch. Compare best tune scores and training time. Different early stopping epochs are part of the reported compute tradeoff. Use full mode and preferably multiple seeds to make strong claims; quick-mode scores are plumbing checks.

Optimization is a separate inference-only experiment: reduce min/max resolution to approximately two-thirds while keeping weights and score/NMS settings fixed. A candidate qualifies if tune mAP50 falls by no more than 0.02 absolute; select the fastest measured median among qualifying policies. If the original is fastest or the candidate fails quality, keep the original. Final validation is inspected only after this choice.

Outputs are generated in each fresh `artifacts/<run>/` directory. No pre-filled numeric results are included.
