//! # YOLO: Real-Time Object Detection
//!
//! You Only Look Once - single-shot object detection
//! Paper: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2015)
//!
//! ## The Object Detection Task
//!
//! **Goal:** Detect and localize multiple objects in an image
//! ```
//! Input: Image
//! Output: [(bbox1, class1, confidence1),
//!          (bbox2, class2, confidence2), ...]
//!
//! Where bbox = (x, y, width, height)
//! ```
//!
//! ## Traditional Approach (R-CNN)
//!
//! ```
//! 1. Region proposal (~2000 regions)
//! 2. Extract features for each region (CNN)
//! 3. Classify each region
//! 4. Refine bounding boxes
//!
//! Problems:
//! • Slow (multiple forward passes)
//! • Complex pipeline
//! • Not end-to-end
//! ```
//!
//! ## YOLO Innovation: Single Forward Pass
//!
//! ```
//! Divide image into S×S grid (e.g., 7×7)
//! Each grid cell predicts:
//! • B bounding boxes (e.g., 2)
//! • Confidence scores
//! • C class probabilities
//!
//! Output tensor: S×S×(B*5+C)
//! Example: 7×7×30 for 2 boxes, 20 classes
//!
//! One CNN forward pass → All predictions!
//! Speed: 45+ FPS (real-time!)
//! ```
//!
//! ## Architecture
//!
//! ```
//! Input: 448×448 image
//!   ↓
//! CNN backbone (24 conv layers)
//!   • Similar to GoogLeNet
//!   • Extract features
//!   ↓
//! Fully connected layers (2 layers)
//!   ↓
//! Output: 7×7×30 tensor
//!   • 7×7 grid
//!   • Each cell: 2 boxes + 20 classes
//! ```
//!
//! ## Predictions Per Grid Cell
//!
//! ```
//! For each cell, predict:
//!
//! Bounding boxes (B=2):
//! • (x, y): Center relative to cell
//! • (w, h): Width/height relative to image
//! • confidence: Pr(Object) * IOU
//!
//! Class probabilities (C=20):
//! • Pr(Class_i | Object)
//!
//! Total: 2*5 + 20 = 30 values per cell
//! ```
//!
//! ## Loss Function
//!
//! ```
//! Multi-part loss:
//!
//! 1. Localization loss (x, y, w, h)
//! 2. Confidence loss (objectness)
//! 3. Classification loss
//!
//! Weighted sum with different λ values
//! ```
//!
//! ## YOLO Versions Evolution
//!
//! **YOLOv1 (2015):** Original, 45 FPS
//! **YOLOv2 (2016):** Batch normalization, anchor boxes, 67 FPS
//! **YOLOv3 (2018):** Multi-scale, better accuracy
//! **YOLOv4 (2020):** CSPDarknet, Mish activation
//! **YOLOv5 (2020):** PyTorch, easy deployment
//! **YOLOv7 (2022):** Trainable bag-of-freebies
//! **YOLOv8 (2023):** Anchor-free, current SOTA
//!
//! ## Applications
//!
//! - Autonomous driving: Real-time vehicle/pedestrian detection
//! - Surveillance: Multiple object tracking
//! - Robotics: Object manipulation
//! - Sports analytics: Player tracking
//! - Retail: Inventory management
//!
//! ## Training Tips
//!
//! ```
//! Pre-training: ImageNet classification
//! Fine-tuning: Detection dataset (COCO, Pascal VOC)
//! Augmentation: Random crops, flips, color jittering
//! Optimizer: SGD with momentum
//! Learning rate: Warmup + decay
//! ```

fn main() {
    println!("=== YOLO: Real-Time Object Detection ===\n");
    println!("You Only Look Once - single-shot detection");
    println!("Speed: 45+ FPS (real-time on GPU)");
    println!("Used in: Autonomous driving, surveillance, robotics");
}
