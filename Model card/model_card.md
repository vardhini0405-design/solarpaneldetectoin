# Solar Panel Detection Model Card

## Model Overview
This model detects rooftop solar panels from high-resolution satellite imagery
using YOLOv8 instance segmentation.

## Training Data
- Source: Roboflow curated rooftop solar dataset
- Imagery: High-resolution satellite images
- Annotation: Polygon masks (COCO segmentation)

## Model Architecture
- YOLOv8 Segmentation
- Input size: 640x640
- Output: Instance masks

## Assumptions
- Clear rooftop visibility
- Daytime satellite imagery
- Urban and semi-urban environments

## Limitations & Bias
- Reduced accuracy under cloud cover
- Lower performance on rural rooftops
- May miss very small installations

## Failure Modes
- False negatives on shadowed rooftops
- Confusion with dark roof materials

## Retraining Guidance
- Add region-specific imagery
- Include seasonal variation
- Increase rural samples
