# Object Detection and the YOLO Family

Object detection combines two tasks: localising objects with bounding boxes and
classifying what each box contains. Detectors are broadly split into two-stage
and single-stage architectures.

Two-stage detectors such as Faster R-CNN first propose candidate regions, then
classify each one. This is accurate but slow, because the classification network
runs once per proposal. Single-stage detectors instead predict boxes and classes
in a single forward pass, trading some accuracy for large speed gains.

YOLO (You Only Look Once) popularised the single-stage approach. The original
YOLOv1 treated detection as direct regression over a grid, running at 45 frames
per second on a Titan X GPU. Later versions steadily closed the accuracy gap
with two-stage methods while remaining real time.

YOLOv8, released by Ultralytics in 2023, uses a CSP-Darknet53 backbone with SiLU
activations, a decoupled detection head, and an anchor-free formulation. Removing
anchors simplifies training because the model no longer needs pre-defined box
priors tuned to the dataset.

YOLOv11, released in 2024, introduces C3k2 blocks, which are cross-stage partial
blocks using smaller 2x2 kernels for finer-grained feature extraction, and C2PSA,
a parallel spatial attention module. Spatial attention lets the network reweight
feature-map regions by task relevance, which particularly helps on small or
partially occluded objects in cluttered scenes.

In practice the choice between versions is a deployment decision rather than a
purely technical one. Larger models detect more but run slower, so citywide
camera networks and edge devices often favour smaller variants, while
high-stakes settings with fewer streams can afford the accuracy.
