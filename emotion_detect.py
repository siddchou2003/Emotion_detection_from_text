import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/emotions.csv")  # Ensure this file exists

# Split data
X = df['text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'models/emotion_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# Predict function
def predict_emotion(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Example usage
if __name__ == "__main__":
    test_text = "I am feeling very low"
    print(f"Input: {test_text}")
    print(f"Predicted Emotion: {predict_emotion(test_text)}")