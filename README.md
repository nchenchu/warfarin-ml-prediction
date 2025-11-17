
# Warfarin Dosage Prediction – Final Machine Learning Project

This repository contains my Final Project for the course BCB 5250 – Machine Learning.  
The goal of this project is to build an end-to-end machine learning pipeline that predicts  
Warfarin dosage categories based on clinical, demographic, and genetic variables.

This project was completed as part of my learning journey.  
I developed the code and analysis myself while receiving guidance and feedback from my professor.  
Some portions of the code structure and troubleshooting steps were supported using AI tools to help me understand concepts better.

---

## Project Objective

Warfarin is a widely used anticoagulant with a very narrow therapeutic window.  
Incorrect dosing can lead to:

- under-dosing → clotting risk 
- Over-dosing → bleeding risk

The purpose of this project is to train machine learning models that can help predict the correct Warfarin dosage category  
for a patient based on the available features.

---

##  Project Structure
warfarin-ml-project/
│
├── src/ # Python scripts for training and prediction
├── notebooks/ # Jupyter notebooks with exploration & deep learning experiments
├── models/ # Trained machine learning models (if included)
├── data/ # Dataset (excluded from GitHub if restricted)
├── docs/ # Project reports, PDFs, deliverables
└── README.md


---

## Machine Learning Methods

This project experiments with several ML models:

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost
- Neural Network (optional notebook)

The models were evaluated using:

- Accuracy  
- Precision & Recall  
- Confusion Matrix  
- ROC–AUC  
- MAE/MSE (for regression trials)

---

## How to Run the Project


---

## Dataset

The project uses a Warfarin dataset containing:  
- Age, weight, height  
- Genotype (VKORC1, CYP2C9)  
- Smoking, alcohol use  
- INR  
- Target dose information  

If the dataset is restricted by course policy, the `data/` folder is excluded via `.gitignore`.

---

## Acknowledgements

This project was completed under the guidance of **Prof. [Your Professor’s Name]**,  
whose feedback helped me understand modeling decisions, data preprocessing techniques,  
and interpret machine learning evaluation metrics.

Some parts of the coding and debugging process were also supported using AI tools  
to help me learn Python syntax, ML concepts, and proper workflow structuring.

---

## Notes

- This project is written in a simple, beginner-friendly style as I am still learning machine learning and Python.  
- The aim is educational — to understand how ML works end-to-end.  
- Future improvements include hyperparameter tuning, SHAP explainability, and model deployment.


