# Source register

External research is distinguished from the proposed experimental design. The register was supplied with the study package (source access cutoff 7 September 2026); entries S03–S07 are 2026 preprints cited as design inspirations only, not as evidence that this study's method works.

| ID | Reference | Role in this study |
|---|---|---|
| S01 | Kaggle. *HMS – Harmful Brain Activity Classification* (2024 competition; data and evaluation description). https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data | Source data, label semantics, offsets, KL evaluation |
| S02 | Sun et al. *An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework* (VIPEEGNet), arXiv 2507.08874, 2025-07-10. https://arxiv.org/abs/2507.08874 | Prior vision-inspired baseline; cohort-scale planning input; scores are not comparable to our split |
| S03 | Kang et al. *DPNeXt: A Lightweight Multi-Scale Feature Fusion Framework for Efficient ViT-Based Multi-Task Dense Prediction*, arXiv 2607.16012, 2026-07-17 (IROS 2026). https://arxiv.org/abs/2607.16012 | Inspiration for economical multi-scale fusion (exploratory P+MSF block) |
| S04 | Hou et al. *Same Brain, Different Prediction: How Preprocessing Choices Undermine EEG Decoding Reliability*, arXiv 2605.07212, 2026-05-08. https://arxiv.org/abs/2605.07212 | Motivates the declared preprocessing-sensitivity condition (STFT window) |
| S05 | Kontras et al. *NeuroAtlas: Benchmarking Foundation Models for Clinical EEG and Brain-Computer Interfaces*, arXiv 2605.14698, 2026-05-14. https://arxiv.org/abs/2605.14698 | Motivates small task-matched controls instead of a foundation-model teacher |
| S06 | Wang et al. *Predicting Only from Selected Evidence: A Tempered Product-of-Experts Bottleneck for Auditable EEG Diagnosis*, arXiv 2608.24377, 2026-08-25. https://arxiv.org/abs/2608.24377 | Motivates deletion/view-drop evidence audits |
| S07 | Mohammadi and Zarei. *RobustSeiz: An Open-Source Framework for Benchmarking the Robustness of EEG Seizure Detection Models*, arXiv 2609.04007, 2026-09-03. https://arxiv.org/abs/2609.04007 | Motivates the prespecified corruption suite |
| S08 | timm model card *mobilenetv3_small_100.lamb_in1k*. https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k | Small ImageNet-pretrained comparator (B3) |
| S09 | NVIDIA DGX Spark documentation and NGC PyTorch release notes (26.08). | Environment policy; CUDA 13.0-era stack retained |
| S10 | NVIDIA. GB10 / Blackwell compute-capability and PyTorch compatibility notes. | Kernel smoke test rationale |
| S11 | Hugging Face Hub documentation: pinned model downloads. | Comparator weight provenance |
| S12 | GitHub documentation: repository and file size limits. | 10 MiB public-file cap, 100 MiB export cap |
| S13 | GitHub documentation: secret scanning and credential incident handling. | Two-scan publication gate |

Additional software: PyTorch 2.9 (CUDA 13.0), timm 1.0.29, NumPy 2.4, SciPy 1.17, pandas 3.0, pyarrow 24, scikit-learn 1.9, matplotlib 3.10, papermill 2.7.
