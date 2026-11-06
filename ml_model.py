import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv(r"C:/Users/Acer/ARYAN CODE/Clg Project/Court_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# Auto columns
name_col = next((c for c in df.columns if "name" in c or "title" in c), df.columns[0])
cat_col = next((c for c in df.columns if "category" in c), df.columns[-1])

df = df.rename(columns={name_col: "name", cat_col: "category"})
df = df.fillna("")

# Features + Target
X = df["name"]
y = df["category"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_vec, y)

# Save
pickle.dump(vectorizer, open("clf_vectorizer.pkl", "wb"))
pickle.dump(model, open("clf_model.pkl", "wb"))

print("✅ Classification Model Trained & Saved")