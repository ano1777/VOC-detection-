# Validation of the delivered source

Performed on CPU, Python 3.12, torch 2.8.0+cpu and torchvision 0.23.0+cpu.
Other checked runtime versions: NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8, Pillow 12.3.0, Streamlit 1.66.1 and nbformat 5.11.1.

Passed:

- Notebook JSON schema validation and syntax checks for every Python module/cell.
- Ten synthetic unit/integration tests (`python -m unittest discover -s tests -v`).
- Actual detector forward/backward, finite loss and classifier gradients.
- Strict checkpoint reload and matching inference outputs.
- One-epoch training-loop execution, tuning evaluation, history and checkpoint creation.
- Actual pretrained COCO weight download, VOC class-row transfer and checkpoint parity.
- CPU latency benchmark execution with valid timing results.
- Streamlit AppTest for missing-model guidance and ready-model startup.
- Every notebook code cell executed in order against a temporary synthetic VOC-shaped fixture, with tiny image limits/resolution. All four experiment runs, comparisons, error analysis, optimization, final evaluation and deployment export completed.

The optional `python tests/smoke_notebook.py` reproduces the synthetic whole-workflow check. It requires the notebook dependencies and downloads pretrained weights if uncached. It uses a temporary directory, changes only in-memory run settings and writes no persistent results. The printed synthetic AP values have no model-quality meaning.

Not performed:

- Training or evaluation on actual PASCAL VOC images.
- CUDA training/AMP verification or GPU timing.
- Manual browser upload and visual inspection of the app with a trained VOC model.
- Multi-seed experiments or production deployment validation.

No synthetic checkpoint or synthetic results are included as trained project deliverables. Complete a full VOC run and the technical report before submission.
