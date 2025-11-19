# 基于 MONAI 的实时结直肠息肉检测与分割 / Real-time Colorectal Polyp CADx with MONAI

## 摘要 Abstract

**中文**  
结直肠息肉的漏检会显著降低结直肠癌的预防效果。我们拟在 MONAI 框架内构建一个面向临床工作流的实时 CADe/CASeg 原型：以 Swin-UNETR/UNETR（ViT 主干）为主模型，辅以 Mamba-UNet（SSM 主干）作为前沿对比；引入 MedSAM/SAM 的 teacher→student 蒸馏提升边界与小目标表现；并通过跨数据集评估与蒸馏/量化实现更好的泛化与接近实时的推理速度。项目使用公开数据（Kvasir-SEG、CVC-ClinicDB、HyperKvasir），完成检测 mAP、分割 Dice/IoU、延时/FPS 等综合评估与误差分析，产出可交互的演示应用与模型/数据卡。本工作旨在兼顾新颖性（ViT/Mamba + SAM 蒸馏）与成功率/可复现性（MONAI 训练管线），并讨论实际部署、合规与商业化路径。

**English**  
Colorectal polyp miss-rates directly impact preventive screening outcomes. We plan to build a real-time CADe/CASeg prototype on top of MONAI: Swin-UNETR/UNETR (ViT backbone) serve as the main models, while Mamba-UNet (SSM backbone) provides an advanced comparison. Teacher→student distillation from MedSAM/SAM sharpens boundaries and small polyps; cross-dataset evaluation plus distillation/quantization improve generalization and near-real-time throughput. Using public datasets (Kvasir-SEG, CVC-ClinicDB, HyperKvasir), we will report detection mAP, segmentation Dice/IoU, latency/FPS, and error analysis, deliver an interactive demo, and publish model/data cards. The goal is to balance novelty (ViT/Mamba + SAM distillation) with reproducibility (MONAI training stack) and explore deployment, compliance, and commercialization opportunities.

---

## 1. 背景与需求 / Background & Need

- **临床痛点 Clinical pain points**: 息肉漏检、边界模糊、医生疲劳、运动模糊、反光/烟雾等。  
  Polyp miss-detection, ambiguous boundaries, clinician fatigue, motion blur, glare, and smoke hamper quality control.
- **监管与转化 Regulatory precedent**: 已有内镜 CADe FDA/CE 先例，证明需求真实且有落地潜力。  
  Existing FDA/CE-cleared endoscopy CADe solutions validate clinical demand and translational potential.
- **资源充足 Data & ecosystem readiness**: 多个公开像素级数据集与成熟的 MONAI/MedSAM 生态，在课程周期内可完成高质量原型。  
  Multiple public pixel-level datasets plus a mature MONAI/MedSAM ecosystem enable a high-quality prototype within one term.

---

## 2. 相关工作简述 / Related Work (Brief)

- **分割 Segmentation**: U-Net/DeepLab → TransUNet, UNETR, Swin-UNETR → Mask2Former-style universal segmentation.
- **检测 Detection**: Anchor-based (YOLO family) → DETR/RT-DETR one-stage transformer detectors.
- **通用分割 Foundation models**: SAM/MedSAM for promptable refinement and pseudo-label creation.
- **前沿 Backbones**: Mamba-UNet, MobileViM leverage selective state space models to balance global context and efficiency.
- **内镜基准 Endoscopy benchmarks**: Kvasir-SEG, CVC-ClinicDB, HyperKvasir remain standard; cross-domain generalization is a key research focus.

---

## 3. 任务公式化 / AI Formulation

- **目标 Objective**: 在保证精度的前提下，实现近实时息肉检测（mAP）+ 分割（Dice/IoU）+ 不确定性估计。  
  Achieve near real-time polyp detection/segmentation with uncertainty estimation while maintaining accuracy.
- **输入 Input**: RGB 内镜帧或短片段，可选光流特征。  
  RGB endoscopy frames or clips, optionally augmented with optical flow.
- **输出 Output**: 候选框、像素级掩膜、置信度/不确定性热图。  
  Bounding boxes, pixel masks, confidence and uncertainty maps.
- **模型 Model stack**:
  - 主线 Mainline: Swin-UNETR / UNETR (ViT encoder + CNN decoder).
  - 对比 Novel: Mamba-UNet or MobileViM-Seg backbones.
  - 蒸馏 Distillation: MedSAM/SAM → student (LoRA/Adapters optional) for sharper boundaries & small polyps.
  - 检测头 Detection head (optional for demo): RT-DETR or Mask2Former-style instance masks.
  - 时序一致性 Temporal consistency: ConvLSTM/TCN smoothing to suppress flicker.
  - 不确定性 Uncertainty: MC-Dropout, deep ensembles, or ECE calibration.

---

## 4. 数据 / Data (Collection, Cleaning, Labelling)

- **来源 Sources**: Kvasir-SEG & CVC-ClinicDB (pixel masks); HyperKvasir (images/video) for external generalization tests.
- **采样与清洗 Sampling & cleaning**: 去重复帧、移除极端模糊/过曝、统一分辨率/色域；合成反光/烟雾/模糊增强。  
  Deduplicate frames, drop extreme blur/overexposure, normalize resolution/colors; add glare/smoke/blur augmentations.
- **划分 Split**: 病人级 Train/Val/Test；设 A→B 外部测试（如 train: Kvasir-SEG, test: CVC-ClinicDB）。  
  Patient-level train/val/test plus external A→B testing (e.g., train on Kvasir-SEG, test on CVC-ClinicDB).
- **数据卡 Data card**: 记录设备、统计分布、偏倚、处理方式与许可声明。  
  Document acquisition devices, statistics, biases, processing, and licensing.

---

## 5. 开发 / Development Logistics

- **软件 Software**: PyTorch 2.x, MONAI (datasets, transforms, inferers, DiceCE/Tversky losses, sliding window), timm; optional ONNX/TensorRT for deployment, Streamlit/Gradio for demos.
- **硬件 Hardware**: 单卡 12–16 GB 显存可完成 2D 训练；若含视频/检测头推荐 ≥24 GB。  
  Single 12–16 GB GPU for 2D tasks; ≥24 GB recommended when adding video/detection heads.
- **参考目录 Suggested layout**:
  ```
  data/
    raw/
    processed/
  configs/
  src/
    data/
    models/
    training/
    inference/
  notebooks/
  demos/
  docs/
  ```
- **时间线 Timeline (5 周示例)**:
  1. **W1**: Baseline Swin-UNETR + data card.
  2. **W2**: Integrate SAM distillation + strong augmentation.
  3. **W3**: Implement Mamba-UNet comparison.
  4. **W4**: External testing, error analysis, lightweighting (distill/quantize/ONNX).
  5. **W5**: Real-time demo, report, and showcase video.

---

## 6. 实验与指标 / Evaluation

- **主指标 Primary**: Dice, IoU (segmentation); mAP@0.5 (detection).
- **效率 Efficiency**: Latency (ms/frame), FPS.
- **泛化 Generalization**: Cross-dataset A→B, stress tests (glare/smoke/blur).
- **不确定性 Uncertainty**: Expected Calibration Error (ECE), threshold-sensitivity curves, low-confidence frame audits.
- **消融 Ablations**:
  - ± SAM distillation.
  - ViT backbone vs. Mamba backbone.
  - ± temporal smoothing.
  - Lightweighting (distill/quantize) impact on accuracy & latency.
- **统计 Statistics**: Patient-level bootstrap 95% CI, paired tests for key improvements.

---

## 7. 预期结果 / Expected Results

- **Kvasir-SEG / CVC-ClinicDB**: Dice ≈ 0.85–0.90 for main models.
- **Cross-domain A→B**: Slight Dice drop but higher than baselines without distillation/augmentation.
- **Inference speed**: ≥20–30 FPS @ 640×640 on a single GPU with AMP + lightweighting.
- *Note*: Actual results depend on training and hardware conditions.

---

## 8. 讨论 / Discussion

- **新颖性 Novelty**: ViT + Mamba backbones, SAM distillation, external generalization, lightweight deployment.
  
- **局限 Limitations**: Public datasets differ from real clinical streams; limited modeling of full video continuity and 3D structure; unified detection+segmentation pipeline requires more engineering.
  
- **缓解 Mitigations**: Multi-dataset training, strong augmentation, external tests, temporal smoothing, and detailed model cards describing applicability boundaries.

---

## 9. 未来计划 / Deployment & Commercialisation

- **部署 Deployment**: Integrate with capture cards/endoscopy workstations; overlay masks & uncertainty in real time; export ONNX/TensorRT for edge devices.
- **合规 Compliance**: Assistive-only workflow, prospective studies, alert-rate monitoring, data/network security policies.
- **商业价值 Business value**: Quality control (PDR/ADR gains), teaching playback, automated reporting.

---

## 10. 风险与伦理 / Risks & Ethics

- **域偏移 Domain shift**: Mitigate via multi-site data, color augmentation, external testing.
- **误报/漏报 False positives/negatives**: Provide uncertainty maps and adjustable thresholds for clinician oversight.
- **伦理 Ethics**: Use public data for research only; label demos as "non-clinical use"; ensure privacy and consent for any future private data.

---

## 11. 分工与进度 / Team & Roles

| 角色 Role | 职责 Responsibility |
| --- | --- |
| PM / 临床接口 PM & Clinical liaison | 需求对接、伦理合规、临床反馈收集 |
| 数据负责人 Data lead | 数据卡、清洗、划分、外部测试安排 |
| 模型-ViT Model lead (ViT) | Swin-UNETR/UNETR 训练与调参 |
| 模型-Mamba Model lead (Mamba) | Mamba-UNet/MobileViM 复现与对比 |
| 蒸馏与加速 Distillation & acceleration | SAM 蒸馏、量化/蒸馏/ONNX、推理优化 |
| 评估与可视化 Evaluation & demo | 指标统计、误差分析、前端 Demo |

---

## 12. 下一步 / Next Steps

1. 搭建数据处理脚本与数据卡模板（含许可/偏倚描述）。
2. 复现 Swin-UNETR 基线，建立训练/验证流水线。
3. 引入 SAM 蒸馏与时序增强，记录提升幅度。
4. 完成 Mamba-UNet/MobileViM 对比实验。
5. 进行跨域测试、误差分析与轻量化部署，最终包装为交互式 Demo。
