import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
import pickle

# Load the dataset
d = pd.read_csv(r"C:\Users\ROHITH\Downloads\Telegram Desktop\indiancrop_dataset.csv")

# Display the first few rows of the dataset
print(d.head())

# Data preprocessing: Check for missing values
print(d.isnull().sum())

# Compute the correlation matrix for numeric columns only
numeric_df = d.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, cmap="crest")
plt.title('Correlation Matrix')
plt.show()  # Display the heatmap first

# Feature selection and target variable
x = d[["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE", "HUMIDITY", "ph", "RAINFALL", "STATE", "CROP_PRICE"]]
y = d.CROP

warnings.filterwarnings("ignore")

# Encode the target variable and 'STATE' column
le = LabelEncoder()
y = le.fit_transform(y)
x["STATE"] = le.fit_transform(d["STATE"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
}

# Dictionary to store accuracy scores
model_scores = {}
predictions = {}

# Train, predict, and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    Pred = model.predict(X_test)
    predictions[model_name] = Pred
    score = accuracy_score(y_test, Pred)
    model_scores[model_name] = score
    print(f"{model_name} Accuracy: {score}")
    print(classification_report(y_test, Pred))

# Plot the actual vs predicted values for each model, one by one
for model_name, Pred in predictions.items():
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
    plt.scatter(range(len(y_test)), Pred, color='red', alpha=0.5, label='Predicted Values')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Crop Category')
    plt.legend()
    plt.show()  # Display and wait for the user to close each plot

# Additional plot: KDE plots for all models, displayed one by one
for model_name, Pred in predictions.items():
    plt.figure(figsize=(10, 5))
    ax = sns.kdeplot(y_test, color="r", label="Actual values")
    sns.kdeplot(Pred, color="b", label="Predicted Value", ax=ax)
    plt.title(f'{model_name} - Actual vs Predicted KDE')
    plt.legend()
    plt.show()  # Display and wait for the user to close each plot

# Additional plot: Bar chart of model accuracy
Algorithms = list(model_scores.keys())
Accuracy = list(model_scores.values())
x_pos = np.arange(len(Accuracy))

plt.bar(x_pos, Accuracy, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(x_pos, Algorithms)
plt.ylabel('Accuracy Score')
plt.xlabel('Machine Learning Models')
plt.title('Model Accuracy Comparison')
plt.show()  # Display the accuracy comparison chart

# Choose the best model based on accuracy
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {model_scores[best_model_name]}")

# Save the best model to a pickle file
pickle.dump(best_model, open("crop_pred.pkl", "wb"))
