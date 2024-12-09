Here is a properly formatted version of the provided content:

---

# CS-643: Cloud Computing - Wine Prediction Assignment

## Objective
This assignment focuses on developing parallel machine learning applications on the **Amazon AWS cloud platform**. The primary goals are:

1. **Utilizing Apache Spark** to train a machine learning model in parallel across multiple EC2 instances.
2. Employing **Spark’s MLlib** for machine learning model development and application in the cloud.
3. Leveraging **Docker** to containerize the ML model, simplifying deployment across environments.

---

## Introduction
The task involves building and deploying a **wine quality prediction model**. This model will be trained using Spark on AWS, utilizing **multiple EC2 instances** for parallel processing. The implementation will be done in **Java** within an **Ubuntu Linux environment**.

---

## Methodology

### Dataset Description
- **TrainingDataset.csv**: Used for training the model on multiple EC2 instances in parallel.
- **ValidationDataset.csv**: Employed for model validation and parameter tuning to optimize performance.

---

### Model Development
- The model is trained using **Spark’s MLlib**, starting with a **logistic regression model** and potentially exploring other models for enhanced performance.
- The **training dataset** is used for model learning, and the **validation dataset** is utilized for performance tuning.
- A **multi-class classification approach** is adopted to classify wine quality scores ranging from **1 to 10**.

---

This structured version improves readability and organization while preserving the core information.
