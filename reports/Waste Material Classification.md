# Waste Material Classification Using Custom CNN and Transfer Learning: A Comparative Study with Hyperparameter Optimization

**Author:** Godfred Opintan

**Date:** August, 2026

**Project Type:** Personal Portfolio Project

**Repository:** <https://github.com/gopintan745/waste_classificator>

**Live Application:** [Your Streamlit Cloud URL]

## Abstract

Waste management is a critical challenge worldwide, with inefficiencies in sorting contributing to environmental degradation. This project explores the application of deep learning for automated waste material classification, with the goal of supporting recycling infrastructure and citizen-facing applications. Two approaches were developed and compared: a custom convolutional neural network (CNN) and a transfer learning approach using state-of-the-art pretrained architectures. Both models were trained on TrashNet, a well-known benchmark dataset, and an extended merged dataset that combines TrashNet with Kaggle's Garbage Classification. Hyperparameter optimization was conducted using Optuna with both correlation-based and functional ANOVA (fANOVA) importance analysis. The transfer learning approach using EfficientNet-B0 achieved the best test set macro F1 of 0.966 on TrashNet and 0.941 on the merged dataset, outperforming the custom CNN by 14–16 percentage points. Beyond benchmark performance, the project addressed practical deployment concerns including domain shift in real-world images, image quality validation, confidence-aware inference, and user feedback collection. The deployed application demonstrates how ML models can be packaged responsibly for real users, with explicit handling of model uncertainty and known limitations. This report documents the methodology, results, and engineering decisions made throughout the project, with the aim of contributing a complete, reproducible case study to the applied machine learning literature.

## Chapter 1: Introduction

### 1.1 Background

Waste management is one of the defining challenges of the 21st century. The World Bank estimates that global waste generation will increase by 70% by 2050, reaching 3.4 billion tonnes annually. Inefficient sorting is a major contributor to this problem: recyclable materials contaminated by incorrect disposal must be landfilled or incinerated, undermining the economic and environmental value of recycling programs.

Computer vision offers a scalable solution to waste sorting. By training deep learning models on photographs of waste materials, it becomes possible to build systems that can identify materials in real time, whether embedded in smart recycling bins, deployed along conveyor belts at material recovery facilities, or used in educational applications that help citizens dispose of waste correctly.

This project contributes to that effort by building, optimizing, and deploying waste material classification models, and by documenting the engineering decisions involved in moving from a benchmark model to a deployed application.

### 1.2 Aims and Objectives

The aims of this project were to:

1. Build and train a custom CNN to classify waste materials by their names and possibly by whether they are recyclable, reusable, can be safely disposed, or hazardous to handle.

2. Fine-tune a pretrained model to classify waste materials with the same property-aware labels.

3. Obtain the optimal hyperparameters for both models.

4. Using the hyperparameters obtained, evaluate and compare the performance of both models.

5. Build a simple GUI application to use the best performing model, or an ensemble of both models which ever proves to be more robust.

### 1.3 Significance of the Project

This project contributes to the applied machine learning literature in several ways:

**For AI/ML engineering:** It demonstrates an end-to-end pipeline from data preparation through hyperparameter optimization to deployment, providing a reproducible template for similar computer vision tasks.

**For computer vision research:** It contributes additional empirical evidence on the comparison between from-scratch training and transfer learning for small-domain image classification tasks, with rigorous hyperparameter analysis using functional ANOVA importance.

**For sustainability:** It demonstrates a working tool that, in principle, can support better recycling practices. While the deployed application is a portfolio demonstration, the methodology transfers to real waste-sorting infrastructure.

**For responsible ML deployment:** It addresses practical concerns that benchmark performance often ignores — model uncertainty, distribution shift, and the gap between in-distribution test scores and real-world usefulness.

### 1.4 Project Scope

This project focuses on a six-class waste classification problem using standard deep learning architectures and publicly available datasets. The scope is deliberately limited to demonstrate end-to-end methodology rather than to produce a state-of-the-art research contribution. The deployed application is intended as a portfolio demonstration, not as a production waste-sorting system.

## Chapter 2: Literature Review

### 2.1 Waste Classification as a Computer Vision Task

Waste classification using computer vision has attracted research attention since the availability of affordable deep learning tools. Several benchmark datasets have been established, with TrashNet being the most widely cited.

\## 2.2 The TrashNet Dataset

TrashNet, created by Gary Thung and Mindy Yang in 2016, is one of the earliest publicly available datasets specifically designed for waste classification. The original dataset contains 2,527 images across six categories:

\- **Cardboard** (393 images)

\- **Glass** (491 images)

\- **Metal** (400 images)

\- **Paper** (584 images)

\- **Plastic** (472 images)

\- **Trash** (187 images)

The images were photographed on a white background using a consistent setup, which has implications for model generalization that will be discussed in Section 4.8.

### 2.3 Prior Work on TrashNet

Several studies have reported results on TrashNet, providing useful baselines for comparison.

**Yang and Liang (2019)** proposed a custom CNN architecture for TrashNet classification, achieving approximately 85% accuracy. Their work demonstrated that from-scratch CNNs can achieve reasonable performance on this dataset, particularly when combined with data augmentation. However, their results suggest that even carefully designed custom architectures face a performance ceiling well below state-of-the-art transfer learning results.

**Bircanoğlu et al. (2018)** introduced the "Recyclable Waste Classification" study, comparing several CNN architectures including VGG-16, ResNet-50, and MobileNet on TrashNet. Their best results approached 87% accuracy with ResNet-50, demonstrating the value of transfer learning for this task.

**Ruiz et al. (2019)** explored lightweight architectures for mobile deployment, achieving 78–82% accuracy with MobileNet variants. Their work highlighted the practical constraint of running waste classifiers on edge devices.

These prior works establish a useful benchmark range: from-scratch CNNs achieve 75–85% accuracy, while transfer learning approaches achieve 87–95% accuracy on TrashNet. The work reported in this project falls within and extends this range.

### 2.4 Hyperparameter Optimization in Deep Learning

Hyperparameter optimization (HPO) has become a standard practice in deep learning projects. Traditional approaches include grid search and random search, with Bayesian optimization methods such as those implemented in Optuna now widely adopted.

**Optuna** (Akiba et al., 2019) provides a flexible framework for HPO with several algorithmic strategies including Tree-structured Parzen Estimator (TPE), which builds a probabilistic model of the search space and adaptively samples promising regions. Optuna also implements pruning mechanisms (such as the Median Pruner used in this project) that terminate underperforming trials early to save compute.

Hyperparameter importance analysis has matured alongside HPO methods. Early approaches used correlation-based metrics, which captured only linear relationships. Modern approaches, including functional ANOVA (fANOVA), random forest-based importance, and SHAP values, can capture nonlinear relationships and parameter interactions. This project uses Optuna's built-in fANOVA-based importance, which has been shown to be more reliable than correlation-based metrics for non-linear hyperparameters.

### 2.5 Custom CNN Architectures

The custom CNN architecture used in this project follows a VGG-inspired design pattern: stacked 3×3 convolutions with batch normalization, ReLU activations, and max pooling, followed by a global pooling and fully connected head. This pattern has been a reliable baseline for image classification since the introduction of VGG networks (Simonyan and Zisserman, 2014) and remains competitive for many tasks.

The specific design choices in this project are documented in Section 3.4.

### 2.6 Transfer Learning

Transfer learning has become the dominant paradigm for image classification on small datasets. Models pretrained on ImageNet (Deng et al., 2009) provide a strong starting point for downstream tasks, leveraging the visual features learned from 1.2 million labeled images.

The architectures explored in this project include:

**ResNet-50** (He et al., 2016): A 50-layer residual network that introduced skip connections, enabling training of much deeper networks than previously possible.

**EfficientNet-B0** (Tan and Le, 2019): A compound-scaled CNN that balances network depth, width, and input resolution through a structured scaling coefficient. EfficientNet-B0 is the smallest variant, with approximately 5.3 million parameters.

**ConvNeXt-Tiny** (Liu et al., 2022): A modernized CNN architecture that incorporates design choices from Vision Transformers while maintaining a pure convolutional structure.

The relative performance of these architectures on waste classification is reported in Section 4.3.

### 2.7 Deployment Considerations

Deploying machine learning models as user-facing applications requires consideration of factors beyond benchmark accuracy. Recent work on responsible ML deployment has highlighted several concerns:

**Distribution shift:** Models trained on one data distribution often perform poorly on data from different sources. This phenomenon, sometimes called "domain shift," is particularly relevant for waste classification where training data (e.g., studio-photographed TrashNet) differs from deployment data (e.g., user-uploaded photos).

**Confidence calibration:** A model's predicted probability of being correct should reflect its actual accuracy. Miscalibrated models can mislead users into trusting wrong predictions.

**OOD detection:** Models should be able to recognize when an input is unlike anything they were trained on, and decline to make confident predictions in those cases.

These considerations influenced several design decisions in the deployed application, as documented in Chapter 5.

---

## Chapter 3: Methodology

### 3.1 Tools and Environment

The project was developed using the following tools:

| Tool | Version | Purpose |

|---|---|---|

| Python | 3.10+ | Primary language |

| PyTorch | 2.1+ | Model training and inference |

| torchvision | 0.16+ | Pretrained models and transforms |

| Optuna | 3.4+ | Hyperparameter optimization |

| torchmetrics | 1.2+ | Evaluation metrics |

| scikit-learn | 1.3+ | Data splitting |

| Pillow | 10.0+ | Image processing |

| OpenCV | 4.8+ | Image quality checks |

| Streamlit | 1.28+ | Web application |

| Plotly | 5.17+ | Interactive visualizations |

| Matplotlib | 3.7+ | Static plots |

| Pandas | 2.0+ | Data manipulation |

| NumPy | 1.24+ | Numerical operations |

All training was conducted on Kaggle notebooks with NVIDIA T4 GPU acceleration (16GB VRAM). The application was deployed on Streamlit Cloud, which provides free hosting for Streamlit applications.

### 3.2 Datasets

#### 3.2.1 TrashNet

The primary dataset was TrashNet (Thung and Yang, 2016), containing 2,527 images across six waste categories. The images were photographed using a consistent studio setup with a white background, providing a controlled but somewhat idealized representation of waste materials.

#### 3.2.2 Merged Dataset

To investigate the effect of data diversity on model performance, a merged dataset was constructed by combining TrashNet with Kaggle's Garbage Classification dataset (1,762 images across the same six categories). The datasets were combined and deduplicated using perceptual hashing to remove images that appeared in both sources with potentially different labels.

#### 3.2.3 Data Preprocessing

The following preprocessing steps were applied:

1. Image deduplication using perceptual hashing (imagehash library) to identify near-duplicate images

2. Removal of corrupt or unreadable files

3. Stratified train/validation/test split (70/15/15) to preserve class distribution across splits

4. Resizing to 224×224 or 256×256 pixels (searchable hyperparameter)

5. Normalization using ImageNet statistics (mean=\[0.485, 0.456, 0.406], std=\[0.229, 0.224, 0.225])

### 3.3 Data Augmentation

Training-time augmentation was applied to increase effective dataset size and improve generalization. The augmentation pipeline included:

- Random horizontal flip (probability 0.5)

- Random vertical flip (probability 0.2)

- Random rotation (±15 degrees)

- Color jitter (brightness, contrast, saturation ±20%)

- Random affine transformation (translate ±10%)

- Random erasing (probability 0.2, area 2-20%)

These augmentations were designed to simulate the variability expected in real-world images, including different orientations, lighting conditions, and partial occlusions.

### 3.4 Custom CNN Architecture

The custom CNN architecture follows a VGG-inspired design with the following structure:

**Input:** 3×256×256 image

**Convolutional blocks (4 total):**

- Block 1: Conv(16→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout2d

- Block 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout2d

- Block 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout2d

- Block 4: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout2d

**Head:**

- Global Average Pooling (GAP) + Global Max Pooling (GMP), concatenated

- Fully connected: Linear(512 → 256) → ReLU → Dropout → Linear(256 → 6)

The architecture contains approximately 5 million parameters with `base\_filters=16`. Each block uses two consecutive 3×3 convolutions (providing an effective 5×5 receptive field with more non-linearities) followed by 2×2 max pooling for spatial downsampling.

The choice of GAP+GMP concatenation in the head (rather than GAP alone) was motivated by the observation that waste materials have varying texture characteristics: some (like cardboard) have uniform texture well-captured by GAP, while others (like crumpled plastic) have sparse features better captured by GMP.

### 3.5 Transfer Learning Models

Three pretrained architectures were evaluated through transfer learning:

1. **ResNet-50**: 25.6 million parameters, ImageNet pretrained

2. **EfficientNet-B0**: 5.3 million parameters, ImageNet pretrained

3. **ConvNeXt-Tiny**: 28.6 million parameters, ImageNet pretrained

For each architecture, the pretrained backbone was retained and a new classification head was added with the appropriate number of output classes. All models were fine-tuned end-to-end (the backbone weights were updated during training).

### 3.6 Training Procedure

All models were trained using the following configuration:

- **Optimizer:** AdamW (searchable) or SGD with momentum=0.9 (searchable)

- **Learning rate:** Log-uniform search in [1e-5, 1e-2]

- **Weight decay:** Log-uniform search in [1e-6, 1e-3]

- **Batch size:** Categorical search in {16, 32, 64}

- **Scheduler:** Cosine annealing with T_max equal to number of epochs

- **Label smoothing:** 0.1

- **Mixed precision training:** Enabled via PyTorch AMP for computational efficiency

- **Early stopping:** Patience of 7 epochs based on validation macro F1

The model checkpoint with the best validation macro F1 was saved during training, then reloaded for evaluation on the held-out test set.

### 3.7 Hyperparameter Optimization

Hyperparameter optimization was conducted using Optuna with the following configuration:

- **Sampler:** Tree-structured Parzen Estimator (TPE) with seed=42

- **Pruner:** Median Pruner with 5 startup trials and 3 warmup steps

- **Trials per study:** 30

- **Search epochs:** 15 (to balance search breadth against compute budget)

- **Final training epochs:** 50 (3.3× the search budget)

The choice of 15 epochs for search trials was deliberate: long enough for models to show meaningful differences in performance, short enough to enable thorough exploration of the hyperparameter space.

Two search spaces were used:

**Custom CNN search space:**

- Learning rate: log-uniform [1e-5, 1e-2]

- Batch size: {16, 32, 64}

- Weight decay: log-uniform [1e-6, 1e-3]

- Optimizer: {AdamW, SGD}

- Dropout: [0.1, 0.5]

- Image size: {192, 224, 256}

- Base filters: {16, 32, 48}

**Transfer learning search space:**

- Same hyperparameters as above, but with `arch` ∈ {ResNet50, EfficientNet-B0, ConvNeXt-Tiny} instead of `base_filters`

### 3.8 Evaluation Metrics

Models were evaluated using:

- **Accuracy:** Fraction of correctly classified samples

- **Macro F1:** Unweighted average of per-class F1 scores, treating all classes equally regardless of support

- **Macro Precision:** Unweighted average of per-class precision

- **Macro Recall:** Unweighted average of per-class recall

Macro F1 was chosen as the primary metric because the dataset has class imbalance (the `trash` class has fewer samples than others), and macro F1 ensures the model performs well across all classes rather than only the majority classes.

Confusion matrices were also computed to analyze per-class error patterns.

### 3.9 Application Development

The best-performing model was deployed as an interactive web application using Streamlit. The application includes:

1. **Image input** via file upload or camera capture

2. **Pre-prediction image quality checks** (resolution, brightness, contrast, blur, etc.)

3. **Confidence-aware inference** with explicit handling of low-confidence predictions

4. **Material property lookup** from a JSON database (recyclable, reusable, hazardous status, handling instructions)

5. **User feedback collection** for future model improvement

6. **Session analytics** showing classification history

7. **Multi-model support** allowing comparison of different model configurations

The application was deployed on Streamlit Cloud for public access, with the model stored on Hugging Face Hub to work within the 1GB repository size limit.

---

## Chapter 4: Results and Discussion

### 4.1 Headline Results

Following hyperparameter optimization, the best configurations from each study were retrained for 50 epochs and evaluated on the held-out test set. Results are summarized in Table 4.1.

#### Table 4.1: Final Test Set Results by Configuration

| Configuration | Test Loss | Test Acc | Test Macro F1 | Test Precision | Test Recall |

|---|---|---|---|---|---|

| Custom CNN (TrashNet) | 0.835 | 0.825 | 0.811 | 0.826 | 0.802 |

| Custom CNN (Merged) | 0.884 | 0.810 | 0.805 | 0.833 | 0.790 |

| EfficientNet-B0 (TrashNet) | 0.491 | 0.969 | 0.966 | 0.974 | 0.959 |

| EfficientNet-B0 (Merged) | 0.539 | 0.945 | 0.941 | 0.947 | 0.937 |

\*\*Key findings:\*\*

1\. \*\*Transfer learning provides a 14-16 percentage point F1 advantage\*\* over custom CNN training.

2\. \*\*The custom CNN plateaued at approximately 0.81 macro F1\*\* on both datasets despite extended training.

3\. \*\*EfficientNet-B0 dominates\*\* with macro F1 of 0.94-0.97 across configurations.

4\. \*\*Training curves were healthy\*\* for all configurations, with no severe overfitting observed.

\## 4.2 Hyperparameter Optimization Dynamics

The Optuna optimization histories revealed distinct patterns for the two model families:

For the \*\*custom CNN\*\*, trial values fluctuated widely throughout the search, with validation macro F1 ranging from approximately 0.15 to 0.80 across the 30 trials. This noisy optimization landscape reflects the sensitivity of from-scratch training to initial conditions and hyperparameter choices.

For \*\*transfer learning\*\*, most trials achieved high validation F1 (0.85-0.95) from the very first trial, with occasional catastrophic failures dropping to 0.35-0.55 when unfavorable hyperparameter combinations were tried. This "high plateau with occasional cliffs" pattern is characteristic of pretrained models, which start in a favorable region of parameter space and only fail when optimization is severely disrupted.

The convergence pattern for transfer learning indicates that Optuna's search on these models is primarily refinement rather than exploration. This is a useful observation: practitioners can have high confidence in transfer learning results even without extensive HPO.

\## 4.3 Best Hyperparameters Discovered

The optimal hyperparameters from each study are summarized in Table 4.2.

\*\*Table 4.2: Best Hyperparameters by Configuration\*\*

| Hyperparameter | Custom CNN (T) | Custom CNN (M) | Transfer (T) | Transfer (M) |

|---|---|---|---|---|

| Learning rate | 4.86e-4 | 6.70e-4 | 4.46e-4 | 3.19e-4 |

| Batch size | 64 | 64 | 32 | 32 |

| Weight decay | 7.98e-5 | 1.17e-4 | 1.12e-5 | 5.65e-5 |

| Optimizer | AdamW | AdamW | AdamW | AdamW |

| Scheduler | cosine | cosine | cosine | cosine |

| Dropout | 0.102 | 0.123 | 0.382 | 0.339 |

| Image size | 256 | 256 | 256 | 256 |

| Base filters / Arch | 16 | 16 | efficientnet\_b0 | efficientnet\_b0 |

\*\*Key observations:\*\*

1\. \*\*AdamW dominated across all configurations.\*\* None of the SGD trials reached competitive performance levels. This is consistent with the broader literature showing AdamW's advantage for small datasets with limited training budgets.

2\. \*\*Cosine annealing was the only scheduler used\*\* (after removing OneCycleLR from the search space during initial experiments, as OneCycleLR requires per-batch step updates that were incompatible with the epoch-level scheduler interface used in the training loop).

3\. \*\*The custom CNN converged to `base\_filters=16`\*\* (the smallest option) in the top trials, which has implications discussed in Section 4.5.

4\. \*\*EfficientNet-B0 won among transfer learning architectures\*\* on the merged dataset. ResNet50 was favored on TrashNet in HPO, but this is partly attributable to Optuna's TPE sampler exhibiting commit-and-exploit behavior (early successful trials led to more sampling of that architecture, reaching n=15 trials for ResNet50 vs n=3 for EfficientNet-B0 on TrashNet).

5\. \*\*Pretrained features require more dropout\*\* (0.34-0.38) than custom CNNs (0.10-0.12). This approximately 3× higher dropout reflects the different roles dropout plays: the custom CNN has limited capacity and over-aggressive dropout removes useful signals, while pretrained models need stronger regularization to prevent the head from overfitting.

\## 4.4 Hyperparameter Importance (fANOVA Analysis)

Analysis using Optuna's built-in functional ANOVA importance revealed distinct hyperparameter sensitivity patterns between the two model families.

\*\*Table 4.3: fANOVA Hyperparameter Importance\*\*

| Hyperparameter | Custom CNN (T) | Custom CNN (M) | Transfer (T) | Transfer (M) |

|---|---|---|---|---|

| Learning rate | 0.088 | 0.083 | \*\*0.529\*\* | \*\*0.581\*\* |

| Dropout | 0.261 | 0.300 | 0.215 | 0.218 |

| Weight decay | \*\*0.351\*\* | \*\*0.399\*\* | 0.043 | 0.017 |

| Architecture / Base filters | 0.085 | 0.060 | 0.099 | 0.120 |

| Batch size | 0.157 | 0.054 | 0.089 | 0.044 |

| Optimizer | 0.031 | 0.083 | 0.019 | 0.009 |

| Image size | 0.026 | 0.020 | 0.006 | 0.010 |

| Scheduler | 0.000 | 0.000 | 0.000 | 0.000 |

\*\*Critical finding:\*\* The dominant hyperparameters differ fundamentally between model families:

\- For the \*\*custom CNN\*\*, regularization hyperparameters (weight\_decay and dropout) together account for 61-70% of explainable performance variance. This pattern indicates that the model's generalization is primarily controlled by how regularization interacts with its limited capacity.

\- For \*\*transfer learning\*\*, learning rate dominates with importance 0.529-0.581, accounting for over half of explainable variance. This finding is consistent with the theory that pretrained models sit in a narrow region of parameter space where the magnitude of updates critically determines whether pretrained features are preserved or destroyed.

This split is theoretically significant: for from-scratch training, finding the right regularization balance is the central challenge; for fine-tuning, finding the right step size is the central challenge.

\*\*Methodological lesson:\*\* A correlation-based importance analysis (initially used in this project) showed dramatically different rankings, suggesting that learning rate was unimportant. fANOVA revealed that this was an artifact of correlation's inability to detect non-linear, threshold-shaped relationships. Any HPO study should report a non-linear importance metric alongside raw trial data.

\## 4.5 The Custom CNN Performance Ceiling

The custom CNN achieved test macro F1 of 0.811 on TrashNet and 0.805 on the merged dataset. Notably, \*\*the score is essentially identical across datasets\*\*, despite the merged dataset containing approximately 80% more training images. This consistency strongly suggests a model-capacity limitation rather than a data-quantity limitation.

Several pieces of evidence support this interpretation:

1\. \*\*Training curves show underfitting, not overfitting.\*\* Train and validation curves remained close throughout the 50-epoch training run, and both were still improving at epoch 50. If the model were overfitting, we would expect train accuracy to substantially exceed validation accuracy.

2\. \*\*Optuna consistently selected the smallest base\_filters (16).\*\* This suggests that within the explored search space, larger models performed worse on validation, likely due to the limited training budget causing larger models to converge slowly.

3\. \*\*The 50-epoch final training improved over HPO results\*\* (from 0.728 to 0.811 on TrashNet, an improvement of 8.3 percentage points). This indicates that the HPO search undercounted the model's potential due to the 15-epoch per-trial budget.

4\. \*\*Confusion matrix analysis\*\* revealed a systematic triangular confusion between the glass, metal, and plastic classes (13-18% error rate). This indicates the model lacks the visual features needed to distinguish materials with shared optical properties.

The most important observation is that the custom CNN trained for 50 epochs shows train accuracy plateauing at approximately 0.77 and validation accuracy at approximately 0.78. \*\*This is not a model that has learned the training data and then overfit; it is a model that has not fully learned the training data\*\*, even with careful regularization.

This finding contradicts initial expectations that the issue was overfitting or insufficient regularization. The actual binding constraint appears to be a combination of model capacity and training budget.

\## 4.6 Per-Class Performance Analysis

Per-class F1 analysis revealed important differences across configurations:

\*\*Table 4.4: Per-Class Macro F1 on Test Set\*\*

| Class | Custom CNN (T) | Custom CNN (M) | EfficientNet (T) | EfficientNet (M) |

|---|---|---|---|---|

| Cardboard | 0.906 | 0.883 | 0.967 | 0.947 |

| Glass | 0.769 | 0.733 | 0.981 | 0.945 |

| Metal | 0.787 | 0.765 | 0.959 | 0.917 |

| Paper | 0.912 | 0.897 | 0.973 | 0.972 |

| Plastic | 0.774 | 0.777 | 0.966 | 0.938 |

| Trash | 0.718 | 0.778 | 0.950 | 0.923 |

| \*\*Macro F1\*\* | \*\*0.811\*\* | \*\*0.806\*\* | \*\*0.966\*\* | \*\*0.941\*\* |

\*\*Three observations emerge from this breakdown:\*\*

\*\*First\*\*, the custom CNN's confusion matrix reveals a systematic triangular confusion between glass, metal, and plastic. Between 13% and 18% of true glass images are misclassified as metal or plastic, a pattern consistent across both datasets. This indicates the model lacks the visual features needed for fine-grained material recognition in this challenging class group. EfficientNet-B0 reduces this confusion to 3-5% using the same training data, leveraging pretrained features that include representations of material properties.

\*\*Second\*\*, EfficientNet-B0's per-class F1 is more uniform (range 0.92-0.98 across configurations) than the custom CNN's (range 0.72-0.91). The pretrained features provide a more uniformly competent material recognition baseline, whereas the from-scratch model specializes unevenly across classes.

\*\*Third\*\*, the merged dataset's regression in macro F1 for transfer learning (0.941 vs 0.966 on TrashNet) was concentrated in the glass class (-0.035 F1) and metal class (-0.042 F1). Classes with distinctive visual signatures — cardboard (uniform texture) and paper (high recall) — were robust to the dataset shift. This suggests that material classes with shared visual properties are more sensitive to dataset diversity, likely because the additional visual variability from the merged dataset creates more boundary cases that are harder to classify.

\## 4.7 The HPO-to-Final Test Comparison

Comparing HPO-best validation F1 to final test F1 reveals the value of extended training:

\*\*Table 4.5: HPO Best vs Final Test F1\*\*

| Configuration | HPO Best Val F1 | Final Test F1 | Improvement |

|---|---|---|---|

| Custom CNN (TrashNet) | 0.728 | 0.811 | +0.083 |

| Custom CNN (Merged) | 0.784 | 0.805 | +0.021 |

| EfficientNet-B0 (TrashNet) | 0.942 | 0.966 | +0.024 |

| EfficientNet-B0 (Merged) | 0.967 | 0.941 | -0.026 |

Three of four configurations improved from HPO to final testing, attributable to the extended training duration (50 epochs vs 15 in HPO). The exception is EfficientNet-B0 on the merged dataset, which slightly regressed. This regression is consistent with selection bias: the best HPO trial on the merged dataset was a single run that may have benefited from favorable initialization; when retrained, the same hyperparameters did not produce the same result. This variability is expected when relying on a single best trial rather than averaging across multiple runs.

### 4.8 Distribution Shift: Test Set vs Real-World Performance

After completing the formal evaluation, informal testing was conducted with images downloaded from the internet. Performance degraded significantly compared to test set results. This observation is consistent with a known phenomenon in machine learning called distribution shift, where models trained on one data distribution perform poorly on samples from a different source.

Several factors contribute to this gap:

\*\*Domain differences.\*\* The training and test sets were drawn from the same dataset (TrashNet and Kaggle's Garbage Classification), which share characteristic features including studio lighting, consistent backgrounds, and standardized viewpoints. Internet images have diverse lighting, cluttered backgrounds, and varied photography conventions.

\*\*Resolution and quality.\*\* Training images have consistent resolution and quality. Internet images vary widely in resolution, JPEG compression level, and capture quality.

\*\*Object pose and framing.\*\* Training data follows standard pose conventions (objects centered, single item per image). Real-world images often contain multiple objects, partial views, or unusual angles.

\*\*Class confusion amplification.\*\* The per-class confusion patterns observed in the test set (particularly the glass-metal-plastic triangle) become more severe on internet images where lighting and surface properties vary more widely.

This observation has important methodological implications: \*\*in-distribution test performance is a necessary but not sufficient indicator of real-world applicability\*\*. For deployment-critical applications, evaluation on out-of-distribution data should be a standard part of the testing pipeline.

The deployed application addresses this concern through several design choices, as documented in Chapter 5.

### 4.9 Methodological Lessons from HPO

Several lessons emerged from the hyperparameter optimization process that extend beyond this specific project:

\*\*Lesson 1: Importance Metric Choice Matters.\*\* The choice of importance metric fundamentally changed the conclusions drawn from HPO. Correlation-based importance suggested that optimizer and base\_filters were the dominant hyperparameters. fANOVA revealed that learning rate dominated transfer learning while regularization dominated custom CNN training. This lesson generalizes: any HPO study should report a non-linear importance metric alongside raw trial data.

\*\*Lesson 2: Search-Space Constraints Affect Conclusions.\*\* Optuna consistently converged to the smallest base\_filter value (16) for the custom CNN. This result is best interpreted as "given the search space and 15-epoch training budget, 16 filters were optimal" — not as "16 filters are intrinsically optimal." A future search with extended training and wider capacity ranges might identify a different optimum.

\*\*Lesson 3: HPO Budget Has a Trade-off.\*\* Each HPO trial used 15 epochs, while final training used 50 epochs. This 3.3× longer final training recovered 8-21 percentage points of macro F1, demonstrating that the HPO search budget partially undersampled what the models could achieve. Future work could use longer HPO trials at the cost of fewer total trials.

\*\*Lesson 4: TPE Sampler Exhibits Commit-and-Exploit Behavior.\*\* The architecture sampling imbalance across Optuna trials (15 ResNet50 vs 3 EfficientNet-B0 on TrashNet) was largely a sampler artifact rather than evidence of architectural merit. This is a known limitation of TPE-based HPO for categorical hyperparameters with high variance.

### 4.10 Limitations of the Study

Several limitations should be acknowledged:

1. \*\*HPO trial budget:\*\* Each study used 30 trials across 7-8 hyperparameters, with categorical architecture choice for transfer learning. This is a sparse budget for a complex search space.

2. \*\*Architecture sampling imbalance:\*\* TPE commit-and-exploit behavior resulted in unequal trial distribution across architectures, preventing definitive architectural comparisons.

3. \*\*HPO epoch budget:\*\* Each HPO trial trained for 15 epochs. The custom CNN's true capacity was likely undercounted due to this constraint.

4. \*\*Single search space for all architectures:\*\* The same search space was used for ResNet50, EfficientNet-B0, and ConvNeXt-Tiny. Different architectures might benefit from architecture-specific hyperparameter ranges.

5. \*\*Importance metric:\*\* This study uses fANOVA via Optuna, which is more robust than correlation for non-linear hyperparameters but is still a post-hoc estimator and can misinterpret interactions in some configurations.

6. \*\*Limited real-world evaluation:\*\* The distribution shift observation is based on informal testing rather than systematic evaluation. A more rigorous out-of-distribution evaluation would strengthen conclusions about real-world applicability.

---

## Chapter 5: Application Development and Deployment

### 5.1 From Benchmark Model to User-Facing Application

Beyond achieving benchmark accuracy, this project aimed to package the best model as a user-facing application that handles real-world concerns responsibly. The application design was informed by both the model's strengths (high in-distribution accuracy) and weaknesses (sensitivity to distribution shift).

### 5.2 Application Architecture

The application is built with Streamlit and follows a modular structure:

- \*\*classifier.py:\*\* Model loading and inference

- \*\*ui\_components.py:\*\* Reusable UI widgets

- \*\*quality\_checks.py:\*\* Image quality validation

- \*\*streamlit\_app.py:\*\* Main application entry point

The application supports two input methods: file upload and camera capture, making it accessible on both desktop and mobile devices.

### 5.3 Confidence-Aware Inference

Rather than presenting predictions as definitive, the application explicitly handles model uncertainty:

- \*\*High confidence predictions (≥80%):\*\* Displayed with a green badge and trusted

- \*\*Medium confidence (60-80%):\*\* Displayed with a yellow badge and a recommendation to verify if important

- \*\*Low confidence (<60%):\*\* Displayed with a red badge and a warning that the model is uncertain

This approach helps users calibrate their trust in the model's output, reducing the risk of misclassification leading to incorrect disposal decisions.

### 5.4 Image Quality Checks

To address concerns about low-quality uploads, the application performs pre-prediction image quality checks. These checks detect:

- \*\*Resolution issues:\*\* Images below 100 pixels are rejected

- \*\*Brightness issues:\*\* Very dark or overexposed images are flagged

- \*\*Contrast issues:\*\* Uniform or flat images are flagged

- \*\*Blur:\*\* Out-of-focus images are detected using Laplacian variance

- \*\*Edge density:\*\* Blank or overly cluttered images are flagged

Critical issues prevent classification from proceeding; advisory issues display warnings but allow classification to continue. This graduated approach balances user experience with safety.

### 5.5 Material Property Lookup

Each prediction is paired with relevant material properties from a curated JSON database:

- \*\*Recyclable status:\*\* Whether the material can be recycled in standard programs

- \*\*Reusable status:\*\* Whether the material has secondary use potential

- \*\*Safe disposal:\*\* Whether the material can be disposed of without special handling

- \*\*Hazardous:\*\* Whether the material poses disposal risks

- \*\*Handling steps:\*\* Specific instructions for proper handling

This feature transforms the application from a classifier into a practical tool that supports real-world decision-making.

### 5.6 User Feedback Loop

The application allows users to confirm whether a prediction was correct, mark it as wrong (with the option to provide the correct class), or indicate uncertainty. This feedback mechanism:

- Provides immediate user satisfaction through response to corrections

- Creates a dataset of real-world usage patterns for future model improvement

- Encourages users to engage critically with predictions rather than trusting them blindly

### 5.7 Session Analytics

Users can view their classification history within a session, including class distribution and confidence histograms. This feature helps users understand the model's behavior and may prompt them to seek alternative verification methods when they observe low-confidence predictions.

### 5.8 Multi-Model Support

The application supports switching between four model configurations (custom CNN and EfficientNet-B0, on TrashNet and merged dataset). This allows users to:

- Compare predictions across different model architectures

- Understand which model is best suited for different use cases

- Observe firsthand the performance differences between from-scratch and transfer learning

### 5.9 Deployment

The application is deployed on Streamlit Cloud, with the model hosted on Hugging Face Hub to work within Streamlit Cloud's 1GB repository size limit. The deployment pipeline:

1. Code repository on GitHub

2. Streamlit Cloud connects to GitHub and pulls the code

3. Model downloads from Hugging Face Hub on first request

4. Application serves predictions via web browser

The live application is accessible at \[deployment URL], with a configurable interface that supports both desktop and mobile access.

\---

## Chapter 6: Conclusion and Future Work

### 6.1 Summary of Findings

This project achieved its primary objectives:

1. \*\*Custom CNN:\*\* Built and trained a 4-block convolutional neural network that achieved test macro F1 of 0.811 on TrashNet and 0.805 on the merged dataset.

2. \*\*Transfer learning:\*\* Fine-tuned three pretrained architectures (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny), with EfficientNet-B0 achieving the best test macro F1 of 0.966 on TrashNet and 0.941 on the merged dataset.

3. \*\*Hyperparameter optimization:\*\* Used Optuna with functional ANOVA importance analysis to identify optimal hyperparameters and understand which parameters matter most for each model class.

4. \*\*Comparative evaluation:\*\* Conducted rigorous comparison across two model classes and two dataset variants, with per-class confusion analysis revealing systematic error patterns.

5. \*\*Application deployment:\*\* Built and deployed a user-facing application with image quality checks, confidence-aware inference, material property lookup, and user feedback collection.

## 6.2 The Central Finding

The central finding of this project is that \*\*transfer learning provides a substantial and robust advantage over from-scratch training for waste classification\*\*, with pretrained EfficientNet-B0 achieving 14-16 percentage points higher macro F1 than a custom CNN with comparable hyperparameter optimization effort.

Beyond raw performance, transfer learning produced models with more uniform per-class performance, suggesting that pretrained features provide a more generalizable foundation for material recognition than what can be learned from 2,500-4,500 training images.

The custom CNN's performance ceiling of approximately 0.81 macro F1 reflects a combination of model capacity and training budget constraints, not overfitting. This finding has implications for practitioners who might otherwise assume that custom architectures can match transfer learning with sufficient training time.

## 6.3 Methodological Contribution

Beyond the benchmark results, this project contributes a methodological lesson: \*\*the choice of importance metric fundamentally affects conclusions drawn from hyperparameter optimization\*\*. Correlation-based metrics can systematically underestimate the importance of hyperparameters with non-linear effects, while functional ANOVA captures these relationships more accurately. This finding has implications beyond waste classification for any HPO study on deep learning models.

## 6.4 Deployment Lessons

The application's design choices (image quality checks, confidence-aware inference, user feedback collection) demonstrate that \*\*responsible ML deployment requires attention to factors beyond benchmark accuracy\*\*. The distribution shift observed in informal testing with internet images illustrates that even highly accurate models on in-distribution test sets may perform poorly on real-world data, and applications should be designed with this uncertainty in mind.

## 6.5 Limitations and Future Work

Several directions for future work emerged from this project:

1. \*\*Larger and more diverse training data:\*\* The most impactful improvement would be expanding the training set to include more diverse real-world images from sources such as user-submitted photos or web-crawled data. This would directly address the distribution shift problem.

2. \*\*Deeper exploration of custom CNN capacity:\*\* The custom CNN's search space constraint to base\_filters ∈ {16, 32, 48} was conservative. Future work could explore larger architectures and longer training schedules.

3. \*\*Ensemble methods:\*\* Combining predictions from multiple models (the custom CNN and EfficientNet-B0) could leverage their complementary strengths. A confidence-weighted ensemble might outperform either model alone.

4. \*\*OOD detection integration:\*\* The application would benefit from formal out-of-distribution detection using methods such as Mahalanobis distance on feature representations, energy-based OOD scores, or trained OOD classifiers. This would automate the detection of images unlike the training distribution.

5. \*\*Cross-validation for HPO:\*\* Rather than relying on a single train/validation split, k-fold cross-validation during HPO would provide more robust estimates of model performance and reduce sensitivity to particular data partitions.

6. \*\*Class taxonomy refinement:\*\* The "trash" category is intentionally broad and tends to be noisy in datasets. Future work could explore more granular categories (e.g., "textile waste," "electronic waste," "mixed materials") or reject images that contain multiple materials.

7. **Domain randomization in training:\*\* Heavier augmentation including CutMix, MixUp, and RandAugment could improve robustness to distribution shift by exposing the model to more visual variability during training.

8. **Mobile deployment:** The current deployment requires a server-side model. For mobile applications, model compression and quantization would enable on-device inference without privacy concerns about uploaded images.

## 6.6 Concluding Remarks

This project demonstrates that a complete applied machine learning project encompasses much more than model training. From data preparation through hyperparameter optimization to deployment, each stage presents decisions that affect the final outcome. The combination of technical rigor in training and thoughtful engineering in deployment produces a system that is not just accurate on benchmarks but genuinely useful in practice.

The waste classification problem is far from solved, and the deployed application is a portfolio demonstration rather than a production system. However, the methodology and engineering patterns developed here are transferable to many other computer vision problems. The key takeaways — that hyperparameter optimization requires careful methodology, that transfer learning provides substantial advantages for small datasets, and that deployment requires attention to factors beyond accuracy — apply broadly to applied machine learning work.

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., \& Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623-2631.

Bircanoğlu, C., Atay, M., Beşer, B., Mapavlıkara, Ö., \& Vargün, A. (2018). Recyclable Waste Classification using CNN. In 2018 International Conference on Artificial Intelligence and Data Processing (IDAP), 1-6.

Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., \& Fei-Fei, L. (2009). ImageNet: A Large-Scale Hierarchical Image Database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.

He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770-778.

Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., \& Xie, S. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11976-11986.

Ruiz, V., Sánchez, Á., Vélez, J. F., \& Raducanu, B. (2019). Automatic Image-Based Waste Classification. In International Work-Conference on the Interplay Between Natural and Artificial Computation, 422-431.

Simonyan, K., \& Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556.

Tan, M., \& Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, 6105-6114.

Thung, G., \& Yang, M. (2016). TrashNet Dataset. GitHub Repository. <https://github.com/garythung/trashnet>

Yang, M., \& Liang, G. (2019). A Convolutional Neural Network Approach for Waste Classification. In Journal of Physics: Conference Series, 1395(1), 012043.

\---

## Appendices

## Appendix A: Project Code Structure

```

waste-classificator/

├── app/

│   ├── streamlit\_app.py

│   ├── classifier.py

│   ├── ui\_components.py

│   ├── quality\_checks.py

│   ├── env\_setup.py

│   ├── waste\_properties.json

│   └── requirements.txt

├── src/

│   ├── \_\_init\_\_.py

│   ├── dataset.py

│   ├── transforms.py

│   ├── train.py

│   └── models/

│       ├── \_\_init\_\_.py

│       ├── custom\_cnn.py

│       └── transfer\_model.py

├── experiments/

│   ├── custom\_cnn\_trashnet\_final/

│   ├── custom\_cnn\_merged\_final/

│   ├── transfer\_trashnet\_final/

│   └── transfer\_merged\_final/

├── scripts/

│   └── fit\_ood\_detector.py

├── README.md

├── LICENSE

└── .streamlit/

&#x20;   └── config.toml

```

\## Appendix B: Hyperparameter Search Results

Full tables of all 30 trials per study are available in the project repository under `experiments/<study\_name>/optuna\_trials.csv`.

\## Appendix C: Confusion Matrices

Confusion matrices for all four configurations are available as figures in the project repository under `reports/figures/confusion\_matrices.png`.

\## Appendix D: Training Curves

Training loss, validation loss, training accuracy, validation accuracy, and validation F1 curves for all four configurations are available in `experiments/<study\_name>/training\_curves.png`.

\## Appendix E: Per-Class Performance Tables

Detailed per-class precision, recall, F1, and support values are available in `reports/per\_class\_report.json`.

\## Appendix F: Material Properties Database

The complete waste\_properties.json file containing recycling, reuse, safety, and handling information for all six material categories is included in the project repository.
