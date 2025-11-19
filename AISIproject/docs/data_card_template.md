# Polyp Dataset Card (Template)

## Overview
- **Name**:  
- **Source/Link**:  
- **License & Usage**: Describe license terms and any restrictions. Include citation format.
- **Modality**: RGB endoscopy frames or clips.
- **Task**: Polyp segmentation (pixel masks) and optional detection.

## Composition
- **Acquisition devices**: Scope model, resolution, frame rate if available.
- **Population/site**: Hospital/site info, patient-level details (if public).
- **Class balance**: Number of images, masks, and proportion of polyp vs. background.
- **Known artifacts**: Motion blur, glare, smoke, specular highlights, clipping.

## Preprocessing
- **Raw location**: `data/raw/<dataset_name>/`
- **Processing steps**: De-duplication, blur/outlier removal, normalization, resizing, mask format.
- **Augmentation (train-time)**: Color jitter, blur, glare/smoke synthesis, flips, rotations.
- **Splits**: Patient-level train/val/test; note any external A→B split.

## Ethical/Privacy Notes
- Public data only; confirm de-identification.
- Declare intended **non-clinical/research** use.
- Document potential biases (site, device, demographic) and applicability limits.

## Label Quality
- Annotation source (clinician? crowd?), mask format, known noise.
- QC checks performed or planned.

## Statistics
- Image/mask counts per split.
- Resolution distribution; brightness/color stats if computed.
- Optional: class-wise coverage, mask area distribution.

## Intended Use & Limits
- Intended: research CADe/CASeg for polyp detection/segmentation.
- Not intended: clinical decision-making without clinician oversight.
- List failure modes: tiny polyps, severe blur, blood, rare devices, extreme lighting.

## Versioning
- **Dataset version/date**:  
- **Processing script hash**: commit or checksum for reproducibility.
- **Change log**: brief notes on updates.
