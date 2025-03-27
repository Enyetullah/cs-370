import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Changed to Classifier
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv("D:/Software Engineering project/workout_fitness_tracker_data.csv")

# One-Hot Encode Categorical Features
categoricalColumns = ['Gender', 'Workout Type', 'Workout Intensity', 'Mood Before Workout', 'Mood After Workout']
dfEncoded = pd.get_dummies(df, columns=categoricalColumns, drop_first=True)

# Define Features (X) and Target (y)
targetColumn = 'Mood After Workout_Neutral'  # Pick a specific one-hot encoded target
X = dfEncoded.drop(columns=[targetColumn])
y = dfEncoded[targetColumn]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
