# Prepare to explain and change the code

Trace one sample from XML to app output:

1. `parse_annotation`: why subtract one from xmin/ymin but not xmax/ymax? Explain inclusive versus half-open coordinates and clipping.
2. `VOCDataset`: what are the shapes/dtypes for zero, one and many objects? Why must horizontal flip also change boxes?
3. `build_model`: what does the backbone/FPN do, then RPN, ROI pooling, classifier and box regressor? Why 21 output classes? How are COCO rows aligned with VOC labels?
4. `train_experiment`: explain four loss terms, SGD, momentum, weight decay, autocast, scaling and clipping order. Why tune mAP rather than training loss for checkpoint selection?
5. `evaluate_detections`: explain IoU, score ranking, greedy matching, duplicate FPs, difficult instances, precision/recall and continuous AP. Why is this IoU-average not official COCO mAP?
6. `benchmark`: why warm up and synchronize CUDA? What is excluded from the measurement? Why does reduced resolution not shrink parameters?
7. `load_checkpoint` and app: why save the class mapping and configuration? What happens if the files are missing? How is normalization consistent?

Try small independent changes and verify them:

- Change one learning rate and add its experiment row.
- Add a photometric augmentation without changing boxes.
- Add filtering by chosen class in the app.
- Export detection counts by class as JSON.
- Add small/medium/large error summaries without mislabeling them as official COCO metrics.
- Diagnose an incorrect bbox flip or swapped coordinate convention using synthetic tests.

Know the limits: no VOC training was run when this source package was generated. Your final defense should use actual training results, observed errors, and changes you understand. A passing smoke test does not show model quality.
