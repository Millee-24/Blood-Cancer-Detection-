🔬 Leukemia Detection SystemLive Demo: 🚀 https://dqptsc5n5ipunqbxwsd53j.streamlit.app/ 🚀 
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

Deployment Overview
This research project has been deployed as a functional web application using Streamlit Cloud, providing real-time diagnostic capabilities based on deep learning.

📂 File Structure & PurposeThe repository is organized to ensure a stable production environment on Python 3.11:
app.py: The main application script that handles the user interface, image preprocessing, and model inference.efficientnetb0_fold_1.h5: The pre-trained weights for our primary model. We utilized Fold 1 specifically because it demonstrated superior clinical specificity, correctly identifying 823 healthy samples during 3-fold cross-validation.requirements.txt: Contains all necessary libraries including tensorflow, opencv-python-headless, and streamlit to ensure the cloud server installs the correct environment.
.python-version: A configuration file that forces the server to use Python 3.11, ensuring compatibility with the TensorFlow wheels required for our 25.0 MB model.

🧪 Model Performance
The deployed EfficientNetB0 model was selected after a comparative study of several CNN architectures:
Metric               EfficientNetB0 (Fold 1)
Mean Accuracy         83.31%
Leukemia Recall       93.0%
Model Size            25.0 MB
