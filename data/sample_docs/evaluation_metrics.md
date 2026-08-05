# Evaluating Detection and Classification Models

Aggregate accuracy is the most commonly reported metric and the most commonly
misleading one. On an imbalanced dataset, a model that always predicts the
majority class can score highly while being useless.

Precision measures what fraction of positive predictions were correct. Recall
measures what fraction of true positives were found. The two trade off against
each other: lowering a confidence threshold catches more true positives but
admits more false ones. Which matters more is a domain question, not a
statistical one. In security screening a missed weapon is far costlier than a
false alarm, so recall dominates. In spam filtering the reverse often holds.

For object detection, mean Average Precision (mAP) is standard. Average
Precision is the area under the precision-recall curve for one class; mAP
averages this across classes. mAP@50 uses a single Intersection over Union
threshold of 0.50, meaning a prediction counts as correct if it overlaps the
ground truth box by at least half. mAP@50-95 averages AP across IoU thresholds
from 0.50 to 0.95 in steps of 0.05, rewarding tighter localisation, and is the
stricter and more informative measure.

Per-class breakdowns matter more than headline figures. A model can post a
strong overall mAP while failing badly on rare classes, and aggregate numbers
hide exactly that. Confusion matrices expose which classes are being confused
with which, often revealing that errors cluster among visually similar
categories rather than being randomly distributed.

ROC-AUC summarises classifier performance across all thresholds and is
threshold-independent, which makes it useful for comparing models. It can be
optimistic on heavily imbalanced data, where precision-recall AUC is a better
guide.
