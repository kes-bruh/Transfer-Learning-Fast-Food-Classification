# Transfer-Learning-Fast-Food-Classification
# Author: Kacin Segovia
## Overview
This project classifies fast-food images using transfer learning with a pretrained ResNet18. Two strategies are compared on the same data and settings: training only a new final layer (frozen backbone) versus fine-tuning every layer.

## Dataset
- **Source:** Hugging Face, `andrewsunanda/fast_food_image_classification`
- **Size:** [3,000 images, 10 classes] (update if you dropped classes or used a subset)
- **Classes:** [list the class names printed by the notebook]
- **Splits:** 70% train / 15% validation / 15% test

The dataset only ships with a `train` split, so the validation and test sets were created in the notebook. A stratified split (`stratify_by_column="label"`) keeps class proportions the same in every split, and a fixed seed (`SEED = 17`) makes the split reproducible.

## Preprocessing and Augmentation
| | Training | Validation / Test |
|---|---|---|
| Resize | RandomResizedCrop to 224x224 (scale 0.6-1.0) | Resize 256, CenterCrop 224 |
| Augmentation | Horizontal flip, RandomErasing (p=0.25) | None |
| Normalization | ImageNet mean and std | ImageNet mean and std |

ImageNet normalization is used because the pretrained model expects it. RandomErasing blocks out random patches so the model cannot rely on one part of the image. Augmentation is applied to training data only, so evaluation stays consistent.

## Model and Training
- **Model:** ResNet18 pretrained on ImageNet, final layer replaced for the number of food classes
- **Framework:** PyTorch Lightning, with torchmetrics for accuracy, macro precision, macro recall and macro F1
- **Optimizer / schedule:** Adam with a cosine learning-rate schedule
- **Training length:** up to 10 epochs, early stopping on validation accuracy (patience 3)
- **Batch size:** 64
- **Checkpointing:** the best checkpoint by validation accuracy is evaluated on the test set (not the last epoch)

| Experiment | Trainable layers | Learning rate |
|---|---|---|
| Frozen backbone | Final layer only | 1e-3 |
| Full fine-tuning | All layers | 1e-4 |

## Results
| Experiment | Val accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|
| Frozen backbone | 0.782 | 0.780 | 0.780 |
| Full fine-tuning | 0.898 | 0.902 | 0.902 |

Full fine-tuning improved test accuracy by about 12 points. Validation and test scores were close in both runs, so the best checkpoint did not appear to be overfit to the validation set.

**Training curves:**
- *Frozen backbone:* training and validation loss and accuracy stayed close together and leveled off around 0.78-0.80 accuracy, so the model was limited by how little of it could learn, not by overfitting.
- *Fine-tuning:* training accuracy reached about 0.99 while validation accuracy plateaued near 0.90, which shows some overfitting. Validation accuracy did not decline, so the selected checkpoint still generalized well.

The notebook also shows example test predictions, with correct and incorrect predictions marked.

## What Was Optimized
- **Data loading speed:** loading the dataset directly was very slow because it is stored as thousands of small image files. Loading the auto-converted Parquet version (`revision="refs/convert/parquet"`) fixed this.
- **Stratified splitting:** keeps class proportions consistent across train, validation and test.
- **Augmentation on training data only:** crops, flips and RandomErasing improve generalization without changing evaluation data.
- **ImageNet normalization:** matches what the pretrained weights expect.
- **Lower learning rate for fine-tuning (1e-4):** protects the pretrained weights from being overwritten too aggressively.
- **Early stopping and best-checkpoint testing:** avoids wasted epochs and reports performance of the best model, not the last one.
- **Reproducibility:** a fixed seed (17) for splits, shuffling and training.

## How to Run
1. Open the notebook in Google Colab and choose a GPU runtime (Runtime > Change runtime type > T4).
2. Run all cells in order (Runtime > Run all). It installs `datasets`, `torchvision`, `lightning` and `torchmetrics`.
3. A Hugging Face token is optional (the dataset is public). If you use one, add it as a Colab secret.

## Limitations
- Each result comes from a single run with one seed, and the test set is only a few hundred images, so small differences should be treated with caution. The 12-point gap is larger than the expected noise.
- Per-class results and a confusion matrix were not analyzed.

## Possible Improvements
- Add a confusion matrix and per-class precision and recall.
- Try unfreezing only the last block, or a larger model such as ResNet50 or EfficientNet.
- Use stronger augmentation or weight decay to reduce overfitting in the fine-tuned model.
- Repeat experiments with several seeds and report the average and spread.
- Test on a dataset with 50 or more visually similar classes (fine-grained classification, such as bird species) to see how the approach handles subtler differences.

## Acknowledgments
- Dataset: `andrewsunanda/fast_food_image_classification` on Hugging Face
- Train/validation/test split reference: https://www.lightly.ai/blog/train-test-validation-split
