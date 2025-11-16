# Object Detection (YOLO) Example

Real-time object detection with YOLO (You Only Look Once).

## Overview

YOLO performs object detection in a single forward pass, enabling real-time performance.

## Running

```bash
cargo run --package object-detection
```

## Key Innovation

```
Traditional: 2000+ region proposals → slow
YOLO: Single pass → 7×7 grid predictions → 45+ FPS
```

## Applications

- Autonomous driving
- Surveillance systems
- Robotics
- Real-time video analysis

## Paper

[You Only Look Once](https://arxiv.org/abs/1506.02640) (Redmon et al., 2015)
