from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from lime.lime_text import LimeTextExplainer
import numpy as np
from urllib.parse import urlparse   

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ✅ TRUSTED WEBSITES (NEW)
# ===============================
trusted_sites = [
    "google.com",
    "youtube.com",
    "chatgpt.com",
    "github.com",
    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "microsoft.com",
    "apple.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "linkedin.com",
    "netflix.com",
    "whatsapp.com",
    "leetcode.com",
    "geeksforgeeks.org",
    "stackoverflow.com",
    "codeforces.com",
    "hackerrank.com"
]

def is_trusted(url):
    domain = urlparse(url).netloc.lower()
    for site in trusted_sites:
        if domain == site or domain.endswith("." + site):
            return True
    return False

# Load models
model_website = joblib.load("xgb_phishing_model.pkl")
vectorizer_website = joblib.load("tfidf_vectorizer.pkl")

model_email = joblib.load("email_model.pkl")
vectorizer_email = joblib.load("email_vectorizer.pkl")

explainer_website = LimeTextExplainer(class_names=["Safe","Phishing"])
explainer_email = LimeTextExplainer(class_names=["Safe","Phishing"])

class URLRequest(BaseModel):
    url: str

class EmailRequest(BaseModel):
    email_text: str

def scale_confidence(probs):
    return [round(float(p)*100,2) for p in probs]


# simple URL analysis for dashboard
def url_analysis(url):
    return {
        "URL Length": len(url),
        "Contains HTTPS": "Yes" if "https" in url else "No",
        "Contains @ Symbol": "Yes" if "@" in url else "No",
        "Contains Hyphen": "Yes" if "-" in url else "No"
    }

@app.post("/predict_website")
def predict_website(data: URLRequest):

    # ===============================
    # ✅ WHITELIST CHECK (ONLY ADD)
    # ===============================
    if is_trusted(data.url):
        return {
            "prediction_label": "Safe",
            "safe_probability": 99.99,
            "phishing_probability": 0.01,
            "confidence": 99.99,
            "reason": "Trusted website (whitelisted)",
            "lime_words": ["trusted", "safe"],
            "lime_values": [-1, -0.8],
            "url_analysis": url_analysis(data.url),
            "theory": "Domain matched with trusted sites list."
        }

    # ===============================
    # 🔽 YOUR ORIGINAL CODE (UNCHANGED)
    # ===============================

    vec = vectorizer_website.transform([data.url])

    probs = model_website.predict_proba(vec)[0]

    safe_prob = round(float(probs[0]) * 100, 2)
    phishing_prob = round(100 - safe_prob, 2)

    confidence = safe_prob

    if confidence < 50:
        prediction_label = "Phishing"
    elif 50 <= confidence <= 60:
        prediction_label = "Not Sure"
    else:
        prediction_label = "Safe"

    def predict_proba(texts):
        return model_website.predict_proba(vectorizer_website.transform(texts))

    lime_exp = explainer_website.explain_instance(data.url, predict_proba, num_features=8)
    lime_list = lime_exp.as_list()

    lime_words = [w for w,_ in lime_list]
    lime_values = [v for _,v in lime_list]

    reason = prediction_label.upper() + " website: " + ", ".join(lime_words)

    return {
        "prediction_label": prediction_label,
        "safe_probability": safe_prob,
        "phishing_probability": phishing_prob,
        "confidence": confidence,
        "reason": reason,
        "lime_words": lime_words,
        "lime_values": lime_values,
        "url_analysis": url_analysis(data.url),
        "theory": "Website detection checks URL patterns (length, HTTPS, IP, @ symbol, hyphens). LIME explains influential tokens."
    }


@app.post("/predict_email")
def predict_email(data: EmailRequest):

    vec = vectorizer_email.transform([data.email_text])

    probs = model_email.predict_proba(vec)[0]

    safe_prob = round(float(probs[0]) * 100, 2)
    phishing_prob = round(100 - safe_prob, 2)

    confidence = safe_prob

    if confidence < 40:
        prediction_label = "Phishing"
    elif 40 <= confidence <= 50:
        prediction_label = "Not Sure"
    else:
        prediction_label = "Safe"

    def predict_proba(texts):
        return model_email.predict_proba(vectorizer_email.transform(texts))

    lime_exp = explainer_email.explain_instance(
        data.email_text,
        predict_proba,
        num_features=10
    )

    lime_list = lime_exp.as_list()

    lime_words = [w for w,_ in lime_list]
    lime_values = [v for _,v in lime_list]

    reason = prediction_label.upper() + " email: " + ", ".join(lime_words)

    return {
        "prediction_label": prediction_label,
        "safe_probability": safe_prob,
        "phishing_probability": phishing_prob,
        "confidence": confidence,
        "reason": reason,
        "lime_words": lime_words,
        "lime_values": lime_values,
        "theory": "Email detection checks suspicious words/links. LIME shows which words contributed to prediction."
    }

@app.get("/")
def home():
    return {"message": "SafeCheck API is running 🚀"}

