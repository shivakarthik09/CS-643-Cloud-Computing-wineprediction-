# CS-643-Cloud-Computing-wineprediction-
Programming Assignment 2: Wine Quality Prediction Model using Apache Spark and AWS
Objective
This assignment focuses on developing parallel machine learning applications on the Amazon AWS cloud platform. The primary goal is to learn:
1.	Utilizing Apache Spark to train a machine learning model in parallel across multiple EC2 instances.
2.	Employing Spark’s MLlib for machine learning model development and application in the cloud.
3.	Leveraging Docker to containerize the ML model, simplifying deployment across environments.
Introduction
The task involves building and deploying a wine quality prediction model. This model will be trained using Spark on AWS, utilizing multiple EC2 instances for parallel processing. The implementation language will be Java on an Ubuntu Linux environment.
Methodology
Dataset Description
•	TrainingDataset.csv: Used for training the model on multiple EC2 instances in parallel.
•	ValidationDataset.csv: Employed for model validation and parameter tuning to optimize performance.
Model Development
•	The model is trained using Spark’s MLlib, starting with a basic logistic regression model, potentially exploring other models for enhanced performance.
•	The model uses the training dataset for learning and the validation dataset for performance tuning.
•	Classification approach considers wine scores from 1 to 10, allowing a multi-class classification model.
