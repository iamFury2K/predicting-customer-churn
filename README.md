# Customer Churn Prediction for a Telecommunication Company

![Telecom](telecom_image.jpg)

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Analysis](#data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Comparison](#model-comparison)
- [Model Performance Metrics](#model-performance-metrics)
- [Conclusion](#conclusion)

## Introduction

This project focuses on predicting customer churn in a telecommunications company using machine learning techniques. Churn prediction is a critical task for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. In this repository, you'll find code and documentation detailing the entire process, from data collection to model development and evaluation, for three different machine learning models: Decision Tree, Logistic Regression, and XGBoost.

## Data Collection

We started by collecting the dataset from Kaggle, which contains historical customer data, including various features such as customer demographics, usage patterns, and churn status.

## Data Analysis

In the data analysis phase, we performed several tasks to understand the dataset:
- Visualized the distribution of churn and non-churn customers using a bar chart.
- Analyzed the relationship between numerical features and churn using box plots.
- Created histograms to explore the distribution of numerical features.
- Examined correlations between numerical features using a heatmap.

## Data Preprocessing

Data preprocessing was crucial to prepare the dataset for modeling:
- Handled missing values by identifying and addressing them appropriately.
- Encoded categorical variables and standardized/normalized numerical features.
- Split the dataset into training and testing sets to train and evaluate our models.
- Removed outliers using the Z-score method.
![ChurnVsFeatures](https://github.com/iamFury2K/predicting-customer-churn/assets/73428754/df1cd70d-0768-40db-908d-31b870212895)
![ChurnVsNoChurnFreq](https://github.com/iamFury2K/predicting-customer-churn/assets/73428754/4d3b1a24-ca18-4f8d-8c7b-30441b341cd5)
## Correlation HeatMap
![correlation_heatmap](https://github.com/iamFury2K/predicting-customer-churn/assets/73428754/feacd2e7-772a-4107-99fc-45023e1dcbd8)
## Histogram
![Histogram with hue](https://github.com/iamFury2K/predicting-customer-churn/assets/73428754/08a16274-fff4-4052-9b9b-04d13a5c2a16)
## Model Development

We developed three machine learning models and evaluated their performance:
1. **Decision Tree Classifier**: We performed hyperparameter tuning and evaluated its performance using metrics such as accuracy, F1 score, precision, recall, Jaccard score, and log loss.
2. **Logistic Regression**: Similar to Decision Tree, we tuned hyperparameters and evaluated model performance. We found the best hyperparameters using grid search.
3. **XGBoost Classifier**: We trained a gradient boosting model and evaluated its performance using the same metrics.

## Model Comparison

To choose the best model, we compared their performance based on accuracy scores. The results are visualized in a horizontal bar chart, showing the accuracy scores of Decision Tree, Logistic Regression, and XGBoost.

## Model Performance Metrics

Here are the performance metrics for each model:

### Decision Tree
- Accuracy Score: 90.7%
- F-1 Score: 0.9070
- Precision Score: 0.9070
- Recall Score: 0.9070
- Jaccard Score: 0.8299
- Log Loss: 3.3504

### Logistic Regression
- Best Hyperparameters: {'C': 10, 'penalty': 'l2', 'random_state': 0}
- Accuracy: 74.5%
- F1 Score: 0.4620
- Precision Score: 0.3395
- Recall Score: 0.7228
- ROC AUC Score: 0.7360

### XGBoost
- Accuracy: 91.6%
- Precision: 0.8814
- Recall: 0.5149
- F1 Score: 0.6500
- ROC AUC Score: 0.7512

## Conclusion

The choice of the best model depends on the specific business goals and the trade-offs between precision and recall. In this analysis, XGBoost achieved the highest accuracy and F1 score, making it a strong candidate for customer churn prediction. However, the final model selection should consider other factors and domain knowledge.

Further model fine-tuning and feature engineering could potentially enhance performance. Additionally, monitoring the model's performance over time and updating it with new data will be essential to maintain its effectiveness in real-world scenarios.
## Accuracy Comparison

![Accuracy_comp](https://github.com/iamFury2K/predicting-customer-churn/assets/73428754/11f59e34-58e4-46fc-bba7-488e0716ff5b)

---

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```python
pip install -r requirements.txt
```
Installing

A step by step series of examples that tell you how to get a development env running.
```
python churn_prediction.py
```
Deployment

Add additional notes about how to deploy this on a live system.
Built With

    Python - The programming language used.
    Scikit-learn - Machine learning library.
    [XGBoost](https://xgboost.readthed
