# Blood-Cancer-Detection
Blood Cancer (ALL) Detection using Deep Learning. This research project performs a comparative analysis of multiple deep learning architectures to detect Acute Lymphoblastic Leukemia (ALL) from microscopic cell images using the C-NMC dataset.

🔬 Research Overview:
The project evaluates four state-of-the-art architectures using 3-fold cross-validation to ensure model robustness and reliable performance metrics:
VGG16: Implemented with manual ImageNet normalization and fixed feature extraction.
EfficientNetB0: Utilizes internal rescaling layers for high computational efficiency.
InceptionV3: Uses a 299 X 299 resolution and label smoothing to improve generalization.
ResNet50: Features a dual-stage pipeline with selective fine-tuning of the top 30 layers.

📊 Performance Metrics
Detailed results, including Confusion Matrices, Learning Curves, and Classification Reports, are organized in the /Results directory for each specific model.

📁 Dataset:
This project utilizes the training subset of the C-NMC 2019 dataset, comprising 10,661 curated microscopic images. The data is evaluated using the original three-fold partitioning (fold_0, fold_1, fold_2) to maintain subject-level independence and ensure robust cross-validation across both ALL and HEM classes.
The models were trained and validated on the C-NMC Leukemia Dataset. Due to storage constraints on GitHub, the raw dataset and trained .h5 model weights are stored externally.


Access the Dataset on Kaggle Here: https://www.kaggle.com/datasets/avk256/cnmc-leukemia
