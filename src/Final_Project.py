import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import gradio as gr
import joblib

# Load Dataset
df = pd.read_excel('warfarin_dataset.xlsx')
print("Columns in the dataset:", df.columns)
features = ['Gender', 'Race','Age', 'Height (cm)', 'Weight (kg)', 'Diabetes', 'Simvastatin', 'Amiodarone', 'Target INR',
            'INR on Reported Therapeutic Dose of Warfarin', 'Cyp2C9 genotypes', 'VKORC1 genotype']

# Target column
target = 'Therapeutic Dose of Warfarin' 

# Data Preprocessing
imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values
df[features] = imputer.fit_transform(df[features])

# Split dataset into features (X) and target (y)
X = df[features]
y = df[target]

X = pd.get_dummies(X, drop_first=True)

# Train-test split (80-20 split for training and testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression Model
regressor_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

regressor_pipeline.fit(X_train, y_train)

y_pred_regressor = regressor_pipeline.predict(X_test)

# Evaluation Metrics
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_regressor))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_regressor))
print("R2 Score:", r2_score(y_test, y_pred_regressor))

# Classification Model
y_class = (y > 30).astype(int)  # High dose (>30 mg/week), low dose (<= 30 mg/week)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

classifier_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

classifier_pipeline.fit(X_train_class, y_train_class)

y_pred_class = classifier_pipeline.predict(X_test_class)

# Evaluation Metrics for Classification
print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("Classification Report:\n", classification_report(y_test_class, y_pred_class))

# Hyperparameter Tuning for Regression Model
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [10, 20, 30],
}

grid_search = GridSearchCV(regressor_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Results Visualization for Regression
plt.scatter(y_test, y_pred_regressor)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Warfarin Doses (Regression)")
plt.show()

# ROC Curve for Classification
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test_class, classifier_pipeline.predict_proba(X_test_class)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Web Application with Gradio
def predict_warfarin_dose(input_data):
    prediction = regressor_pipeline.predict([input_data])
    return prediction[0]

interface = gr.Interface(fn=predict_warfarin_dose, 
                         inputs=[gr.Textbox(label='Input data as comma-separated values')], 
                         outputs='text')

interface.launch()

# Save Model
joblib.dump(regressor_pipeline, 'warfarin_dose_predictor_model.pkl')

# Load Model
model = joblib.load('warfarin_dose_predictor_model.pkl')
