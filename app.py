
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===== DEEP LEARNING IMPORTS =====
import torch
from sentence_transformers import SentenceTransformer
from model_def import OutcomeClassifier

# ===== GEMINI + ENV IMPORTS =====
from google import genai
from dotenv import load_dotenv

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "Court_dataset.csv")

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None
GEMINI_MODEL_NAME = "gemini-3.6-flash"

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[INFO] Gemini API key loaded successfully.")
    except Exception as e:
        print(f"[GEMINI INIT ERROR] {e}")
else:
    print("[WARN] GEMINI_API_KEY not set. /analyze_case will not generate narratives.")

# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)

# ==========================================
# GLOBAL CACHE
# ==========================================

df_global = None
vectorizer_global = None
matrix_global = None

model_global = None
le_type_global = None
le_cat_global = None
le_out_global = None

embed_model_global = None
dl_model_global = None
dl_label_encoder_global = None

# ==========================================
# LOAD DATASET
# ==========================================

def load_data():

    global df_global

    if df_global is not None:
        return df_global

    if not os.path.exists(DATA_PATH):

        print(f"[ERROR] Dataset not found at: {DATA_PATH}")

        df_global = pd.DataFrame(
            columns=[
                "name",
                "case_type",
                "case_category",
                "judge",
                "status",
                "outcome",
                "year"
            ]
        )

        return df_global

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip().str.lower()

    df = df.loc[:, ~df.columns.duplicated()]

    df = df.fillna("")

    # Extract year from case name
    if "year" not in df.columns:
        df["year"] = df["name"].str.extract(
            r'(\d{4})$'
        ).fillna("")
    else:
        extracted_year = df["name"].str.extract(
            r'(\d{4})$'
        )[0]

        df["year"] = extracted_year.fillna(
            df["year"].astype(str)
        )

    # Convert columns to string
    df["name"] = df["name"].astype(str)
    df["case_type"] = df["case_type"].astype(str)
    df["case_category"] = df["case_category"].astype(str)

    df["judge"] = (
        df["judge"].astype(str)
        if "judge" in df.columns else ""
    )

    df["status"] = (
        df["status"].astype(str)
        if "status" in df.columns else ""
    )

    df["outcome"] = (
        df["outcome"].astype(str)
        if "outcome" in df.columns else ""
    )

    # Combined text for search
    df["combined"] = (
        df["name"] + " " +
        df["case_type"] + " " +
        df["case_type"] + " " +
        df["case_category"] + " " +
        df["judge"] + " " +
        df["status"] + " " +
        df["outcome"] + " " +
        df["year"].astype(str)
    )

    df_global = df

    print(f"[INFO] Dataset loaded: {len(df)} rows")

    return df


# ==========================================
# TF-IDF SEARCH MODEL
# ==========================================

def load_or_train_search_model(df):

    global vectorizer_global, matrix_global

    if vectorizer_global is not None:
        return vectorizer_global, matrix_global

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(df["combined"])

    vectorizer_global = vectorizer
    matrix_global = matrix

    print("[INFO] TF-IDF search model trained.")

    return vectorizer, matrix


def ml_search(query, df, vectorizer, matrix):

    if not query.strip():
        return df.head(20)

    query_vec = vectorizer.transform([query])

    similarity = cosine_similarity(
        query_vec,
        matrix
    ).flatten()

    top_indices = similarity.argsort()[-50:][::-1]

    results = df.iloc[top_indices].copy()

    results["score"] = similarity[top_indices]

    results = results[results["score"] > 0]

    return results


# ==========================================
# RANDOM FOREST PREDICTION MODEL
# ==========================================

def train_prediction_model(df):

    global model_global
    global le_type_global
    global le_cat_global
    global le_out_global

    if model_global is not None:
        return (
            model_global,
            le_type_global,
            le_cat_global,
            le_out_global
        )

    df = df.copy()

    df = df[
        df["outcome"].notna()
        & (df["outcome"] != "")
        & (df["outcome"].str.lower() != "nan")
    ]

    if len(df) == 0:
        raise ValueError("No valid outcome data available.")

    le_type = LabelEncoder()
    le_cat = LabelEncoder()
    le_out = LabelEncoder()

    df["type_enc"] = le_type.fit_transform(
        df["case_type"]
    )

    df["cat_enc"] = le_cat.fit_transform(
        df["case_category"]
    )

    df["out_enc"] = le_out.fit_transform(
        df["outcome"]
    )

    X = df[["type_enc", "cat_enc"]]

    y = df["out_enc"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    model_global = model
    le_type_global = le_type
    le_cat_global = le_cat
    le_out_global = le_out

    print("[INFO] RandomForest prediction model trained.")

    return model, le_type, le_cat, le_out


# ==========================================
# DEEP LEARNING MODEL LOADER
# ==========================================

def load_dl_model():

    global embed_model_global
    global dl_model_global
    global dl_label_encoder_global

    if dl_model_global is not None:

        return (
            embed_model_global,
            dl_model_global,
            dl_label_encoder_global
        )

    config_path = os.path.join(
        BASE_DIR,
        "dl_model_config.pkl"
    )

    weights_path = os.path.join(
        BASE_DIR,
        "dl_model.pt"
    )

    label_path = os.path.join(
        BASE_DIR,
        "dl_label_encoder.pkl"
    )

    if not (
        os.path.exists(config_path)
        and os.path.exists(weights_path)
        and os.path.exists(label_path)
    ):

        print(
            "[WARN] Deep learning model files not found. "
            "Run train_dl_model.py first."
        )

        return None, None, None

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    with open(label_path, "rb") as f:
        le_out = pickle.load(f)

    embed_model = SentenceTransformer(
        config["embed_model_name"]
    )

    dl_model = OutcomeClassifier(
        input_dim=config["input_dim"],
        hidden_dim=128,
        num_classes=config["num_classes"]
    )

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True
    )

    dl_model.load_state_dict(state_dict)

    dl_model.eval()

    embed_model_global = embed_model
    dl_model_global = dl_model
    dl_label_encoder_global = le_out

    print("[INFO] Deep learning model loaded.")

    return embed_model, dl_model, le_out


# ==========================================
# GEMINI AI CASE NARRATIVE GENERATION
# ==========================================

def generate_case_narrative(case_row):

    """
    Generates an illustrative AI case analysis
    based on available case metadata.

    This is not a verified legal judgment or
    an actual case-document summary.
    """

    if gemini_client is None:

        return (
            "AI narrative unavailable. "
            "Gemini API is not configured. "
            "Check your .env file and restart the server."
        )

    prompt = f"""
You are a legal analyst AI helping a law and computer
science student understand court cases.

Based on the following case metadata, write a clear,
well-structured illustrative case analysis
of around 150-200 words.

Cover:

1. What the case may be about.
2. Possible arguments of each side.
3. Relevant legal considerations.
4. Factors that could influence the outcome.

IMPORTANT:
Do not invent verified facts, specific legal citations,
or actual court findings.

Clearly mention that this is an illustrative analysis
based only on available metadata.

Case Name: {case_row['name']}
Case Type: {case_row['case_type']}
Case Category: {case_row['case_category']}
Presiding Judge: {case_row['judge']}
Status: {case_row['status']}
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )

        if response.text:
            return response.text.strip()

        return "Gemini returned an empty response."

    except Exception as e:

        print(f"[GEMINI ERROR] {e}")

        return "AI narrative generation failed. Please check the API configuration and server logs."


# ==========================================
# CASE LEVEL INSIGHTS
# ==========================================

def compute_case_insights(case_row, df):

    category = case_row["case_category"]

    judge = case_row["judge"]

    valid_outcomes = df[
        df["outcome"].isin(
            ["Won", "Lost", "Settled"]
        )
    ]

    def outcome_breakdown(sub_df):

        total = len(sub_df)

        if total == 0:

            return {
                "total": 0,
                "won_pct": 0,
                "lost_pct": 0,
                "settled_pct": 0
            }

        counts = sub_df["outcome"].value_counts()

        return {
            "total": total,
            "won_pct": round(
                counts.get("Won", 0) / total * 100,
                1
            ),
            "lost_pct": round(
                counts.get("Lost", 0) / total * 100,
                1
            ),
            "settled_pct": round(
                counts.get("Settled", 0) / total * 100,
                1
            )
        }

    cat_df = valid_outcomes[
        valid_outcomes["case_category"] == category
    ]

    judge_df = valid_outcomes[
        valid_outcomes["judge"] == judge
    ]

    return {
        "category": category,
        "judge": judge,
        "category_stats": outcome_breakdown(cat_df),
        "judge_stats": outcome_breakdown(judge_df)
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# LIVE CASE SEARCH
# ==========================================

@app.route("/live_search")
def live_search():

    query = request.args.get("q", "").strip()

    df = load_data()

    if df.empty:
        return jsonify([])

    vectorizer, matrix = load_or_train_search_model(df)

    results = ml_search(
        query,
        df,
        vectorizer,
        matrix
    )

    cols = [
        "name",
        "case_type",
        "case_category",
        "judge",
        "status",
        "outcome",
        "year",
        "score"
    ]

    cols = [
        c for c in cols
        if c in results.columns
    ]

    output = results[cols].copy()

    return jsonify(
        output.to_dict(orient="records")
    )


# ==========================================
# RECENT CASES BY CATEGORY
# ==========================================

@app.route("/recent_by_category")
def recent_by_category():

    category = request.args.get(
        "category",
        ""
    ).strip()

    if not category:
        return jsonify([])

    df = load_data()

    filtered = df[
        df["case_category"].str.lower()
        == category.lower()
    ].copy()

    if filtered.empty:

        filtered = df[
            df["case_category"].str.lower().str.contains(
                re.escape(category.lower()),
                na=False
            )
        ].copy()

    filtered["year_num"] = pd.to_numeric(
        filtered["year"],
        errors="coerce"
    ).fillna(0)

    filtered = filtered.sort_values(
        "year_num",
        ascending=False
    )

    cols = [
        "name",
        "case_type",
        "case_category",
        "judge",
        "status",
        "outcome",
        "year"
    ]

    cols = [
        c for c in cols
        if c in filtered.columns
    ]

    return jsonify(
        filtered[cols].head(30).to_dict(
            orient="records"
        )
    )


# ==========================================
# RANDOM FOREST PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    case_type = request.form.get(
        "case_type",
        ""
    ).strip()

    case_category = request.form.get(
        "case_category",
        ""
    ).strip()

    if not case_type or not case_category:

        return jsonify({
            "prediction": "Enter valid inputs"
        })

    df = load_data()

    try:

        model, le_type, le_cat, le_out = (
            train_prediction_model(df)
        )

        case_type = case_type.title()

        case_category = case_category.title()

        type_enc = le_type.transform(
            [case_type]
        )[0]

        cat_enc = le_cat.transform(
            [case_category]
        )[0]

        pred = model.predict(
            [[type_enc, cat_enc]]
        )[0]

        result = le_out.inverse_transform(
            [pred]
        )[0]

        return jsonify({
            "prediction": result
        })

    except ValueError as e:

        print(f"[PREDICT ERROR] {e}")

        valid_types = (
            list(le_type_global.classes_)
            if le_type_global is not None else []
        )

        valid_cats = (
            list(le_cat_global.classes_)
            if le_cat_global is not None else []
        )

        return jsonify({
            "prediction": "Unknown",
            "valid_types": valid_types,
            "valid_categories": valid_cats
        })

    except Exception as e:

        print(f"[PREDICT ERROR] {e}")

        return jsonify({
            "prediction": "Prediction failed"
        }), 500


# ==========================================
# DEEP LEARNING PREDICTION
# ==========================================

@app.route("/predict_dl", methods=["POST"])
def predict_dl():

    case_description = request.form.get(
        "case_description",
        ""
    ).strip()

    if not case_description:

        return jsonify({
            "prediction": "Enter a case description"
        })

    try:

        embed_model, dl_model, le_out = load_dl_model()

        if dl_model is None:

            return jsonify({
                "prediction": (
                    "DL model not trained yet. "
                    "Run train_dl_model.py first."
                )
            })

        embedding = embed_model.encode(
            [case_description]
        )

        embedding_tensor = torch.tensor(
            embedding,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = dl_model(embedding_tensor)

            pred_idx = torch.argmax(
                output,
                dim=1
            ).item()

            confidence = torch.softmax(
                output,
                dim=1
            )[0][pred_idx].item()

        result = le_out.inverse_transform(
            [pred_idx]
        )[0]

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 3)
        })

    except Exception as e:

        print(f"[DL PREDICTION ERROR] {e}")

        return jsonify({
            "prediction": "Deep learning prediction failed"
        }), 500


# ==========================================
# COMPLETE AI CASE ANALYSIS
# ==========================================

@app.route("/analyze_case", methods=["POST"])
def analyze_case():

    case_name = request.form.get(
        "case_name",
        ""
    ).strip()

    if not case_name:

        return jsonify({
            "error": "No case selected"
        })

    df = load_data()

    match = df[
        df["name"] == case_name
    ]

    if match.empty:

        return jsonify({
            "error": "Case not found"
        }), 404

    case_row = match.iloc[0]

    # 1. Gemini AI narrative
    narrative = generate_case_narrative(
        case_row
    )

    # 2. Deep learning prediction
    embed_model, dl_model, le_out = load_dl_model()

    prediction = None
    confidence = None

    if dl_model is not None:

        combined_text = (
            f"{case_row['name']} "
            f"{case_row['case_type']} "
            f"{case_row['case_category']} "
            f"{narrative}"
        )

        embedding = embed_model.encode(
            [combined_text]
        )

        embedding_tensor = torch.tensor(
            embedding,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = dl_model(
                embedding_tensor
            )

            pred_idx = torch.argmax(
                output,
                dim=1
            ).item()

            confidence = torch.softmax(
                output,
                dim=1
            )[0][pred_idx].item()

        prediction = le_out.inverse_transform(
            [pred_idx]
        )[0]

    return jsonify({

        "case": {
            "name": case_row["name"],
            "case_type": case_row["case_type"],
            "case_category": case_row["case_category"],
            "judge": case_row["judge"],
            "status": case_row["status"],
            "year": case_row.get("year", "")
        },

        "narrative": narrative,

        "prediction": prediction,

        "confidence": (
            round(confidence, 3)
            if confidence is not None
            else None
        ),

        "insights": compute_case_insights(
            case_row,
            df
        )

    })


# ==========================================
# ADVANCED ANALYTICS DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    df = load_data()

    total = len(df)

    types = (
        df["case_type"].value_counts().head(5).to_dict()
        if "case_type" in df.columns else {}
    )

    categories = (
        df["case_category"].value_counts().head(5).to_dict()
        if "case_category" in df.columns else {}
    )

    pending = (
        len(
            df[df["status"].str.lower() == "pending"]
        )
        if "status" in df.columns else 0
    )

    closed = (
        len(
            df[df["status"].str.lower() == "closed"]
        )
        if "status" in df.columns else 0
    )

    judges = (
        df["judge"].value_counts().head(10).to_dict()
        if "judge" in df.columns else {}
    )

    # Outcome distribution
    outcomes = (
        df[
            df["outcome"].isin(
                ["Won", "Lost", "Settled"]
            )
        ]["outcome"].value_counts().to_dict()
    )

    total_resolved = sum(
        outcomes.values()
    )

    win_rate = (
        round(
            outcomes.get("Won", 0)
            / total_resolved * 100,
            1
        )
        if total_resolved else 0
    )

    # Year-wise case trend
    year_df = df[
        df["year"].astype(str).str.strip() != ""
    ].copy()

    year_df["year"] = pd.to_numeric(
        year_df["year"],
        errors="coerce"
    )

    year_df = year_df.dropna(
        subset=["year"]
    )

    year_counts = (
        year_df.groupby("year")
        .size()
        .sort_index()
    )

    years = [
        str(int(y))
        for y in year_counts.index.tolist()
    ]

    year_values = (
        year_counts.values.tolist()
    )

    return render_template(
        "dashboard.html",
        total=total,
        types=types,
        categories=categories,
        judges=judges,
        pending=pending,
        closed=closed,
        outcomes=outcomes,
        win_rate=win_rate,
        years=years,
        year_values=year_values
    )


# ==========================================
# GET CASE TYPE AND CATEGORY OPTIONS
# ==========================================

@app.route("/get_options")
def get_options():

    df = load_data()

    types = sorted(
        df["case_type"].unique().tolist()
    )

    categories = sorted(
        df["case_category"].unique().tolist()
    )

    return jsonify({
        "case_types": types,
        "case_categories": categories
    })


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n===================================")
    print(" LEGAL INTELLIGENCE SYSTEM")
    print("===================================")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
