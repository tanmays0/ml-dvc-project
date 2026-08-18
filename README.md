# Reproducible ML Experiment Pipeline using Git and DVC

## 1. Project Overview

This project demonstrates a reproducible machine-learning experiment pipeline using Git and DVC.

The Iris dataset is used to train a Random Forest classification model. Git is used for version control of source code, parameters, and pipeline configuration, while DVC is used for dataset and machine-learning pipeline management.

The project contains three pipeline stages:

1. Data preprocessing
2. Model training
3. Model evaluation

The pipeline can be reproduced using DVC with a single command.

---

## 2. Problem Statement

Develop a machine-learning model using a dataset and create a reproducible experiment pipeline using Git and DVC.

The project demonstrates that different versions of model parameters can be tracked and reproduced while maintaining experiment history.

---

## 3. Dataset

The project uses the Iris dataset.

The dataset contains 150 samples and four numerical features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target contains three Iris flower classes:

- Setosa
- Versicolor
- Virginica

The raw dataset is tracked using DVC.

---

## 4. Technologies Used

- Python
- Pandas
- Scikit-learn
- PyYAML
- Git
- GitHub
- DVC
- Random Forest Classifier

---

## 5. Project Structure

```text
ml-dvc-project/
│
├── data/
│   ├── dataset.csv
│   ├── dataset.csv.dvc
│   └── processed.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── model.pkl
│
├── metrics/
│   └── metrics.json
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── .gitignore