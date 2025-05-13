# Deep-learning-for-Ai-application-
import pandas as pd

try:
    df = pd.read_csv('DataB.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'DataB.csv' not found.")
    df = None
except pd.errors.ParserError:
    print("Error: Could not parse 'DataB.csv'. Check the file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None
    # Data Shape and Info
print("DataFrame Shape:", df.shape)
df.info()

# Descriptive Statistics
print("\nDescriptive Statistics:")
display(df.describe())

# Missing Value Analysis
print("\nMissing Value Analysis:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("Missing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)

# Target Variable Analysis
print("\nTarget Variable Analysis:")
print("Target Variable Value Counts:\n", df['gnd'].value_counts())
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
df['gnd'].value_counts().plot(kind='bar')
plt.title('Distribution of Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()

# Correlation Analysis (only for numerical features)
print("\nCorrelation Analysis (Numerical Features):")
numerical_features = df.select_dtypes(include=['number'])
correlation_matrix = numerical_features.corr()
display(correlation_matrix)

# Data Type Examination
print("\nData Type Examination:")
print(df.dtypes)
import matplotlib.pyplot as plt

# Investigate 'Unnamed: 0'
plt.figure(figsize=(8, 6))
plt.scatter(df['Unnamed: 0'], df['gnd'], alpha=0.5)
plt.title('Unnamed: 0 vs. gnd')
plt.xlabel('Unnamed: 0')
plt.ylabel('gnd')
plt.show()

print("Unique values in 'Unnamed: 0':", df['Unnamed: 0'].unique())
# 4. Prepare data for splitting
# Select the final set of features and the target variable
final_features = list(df.columns)
final_features.remove('gnd')
X = df[final_features]
y = df['gnd']

print("Number of features selected:", len(final_features))
print("Final features used for modeling:\n", final_features)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
