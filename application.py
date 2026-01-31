# app.py
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import joblib
from config.paths_config import MODEL_OUTPUT_PATH, CONFIG_PATH
from utils.common_functions import read_yaml

app = Flask(__name__)

# ===== load model and config once =====
model = joblib.load(MODEL_OUTPUT_PATH)
config = read_yaml(CONFIG_PATH)
feature_names = getattr(model, "feature_names_in_", None)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        # ---------- 1. Read raw inputs from form ----------
        gender = request.form["gender"]              # 'F' / 'M'
        age = int(request.form["age"])

        scholarship = int(request.form["scholarship"])
        hipertension = int(request.form["hipertension"])
        diabetes = int(request.form["diabetes"])
        alcoholism = int(request.form["alcoholism"])
        handcap = int(request.form["handcap"])
        sms_received = int(request.form["sms_received"])

        neighbourhood = request.form["neighbourhood"]
        date_diff = float(request.form["date_diff"])

        # ---------- 2. Build 1-row raw DataFrame ----------
        df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Scholarship": scholarship,
            "Hipertension": hipertension,
            "Diabetes": diabetes,
            "Alcoholism": alcoholism,
            "Handcap": handcap,
            "SMS_received": sms_received,
            "Neighbourhood": neighbourhood,
            "Date.diff": date_diff,
            # Showed_up not known at prediction time
        }])

        # ---------- 3. Feature engineering ----------
        # Age_capped
        df["Age_capped"] = df["Age"].clip(upper=100)

        # Any_condition & Has_condition
        df["Any_condition"] = (
            df["Hipertension"].astype(int)
            + df["Diabetes"].astype(int)
            + df["Alcoholism"].astype(int)
            + df["Handcap"].astype(int)
        )
        df["Has_condition"] = (df["Any_condition"] > 0).astype(int)

        # Neighbourhood_grouped
        top_Neighbourhood = config["data_processing"]["top_Neighbourhood"]
        # In training you used value_counts().nlargest(...); here we just assume
        # the config also stores that list, or you hard‑code it.
        # If you saved the list in config, load it:
        top_neigh = config["data_processing"].get("top_neigh_list", [])
        if top_neigh:
            df["Neighbourhood_grouped"] = df["Neighbourhood"].where(
                df["Neighbourhood"].isin(top_neigh),
                other="Other"
            )
        else:
            # Fallback: treat anything not equal "Other" as itself, else "Other"
            df["Neighbourhood_grouped"] = df["Neighbourhood"]

        # Date_diff_log
        df["Date_diff_log"] = np.log1p(df["Date.diff"])

        # Drop original columns you dropped in training
        drop_cols = ["Age", "Any_condition", "Neighbourhood", "Date.diff"]
        df_model = df.drop(columns=drop_cols)

        # Encoding
        df_model_enc = df_model.copy()
        df_model_enc["Gender"] = df_model_enc["Gender"].map({"F": 0, "M": 1}).astype(int)

        bool_cols = [
            "Scholarship", "Hipertension", "Diabetes",
            "Alcoholism", "Handcap", "SMS_received", "Has_condition"
        ]
        df_model_enc[bool_cols] = df_model_enc[bool_cols].astype(int)

        df_model_enc = pd.get_dummies(
            df_model_enc,
            columns=["Neighbourhood_grouped"],
            drop_first=True
        )

        # ---------- 4. Align with model feature order ----------
        X = df_model_enc
        if feature_names is not None:
            X = X.reindex(columns=feature_names, fill_value=0)

        # ---------- 5. Predict ----------
        proba = model.predict_proba(X)[0, 1]
        prediction = int(proba >= 0.5)   # 1 = likely no‑show
        probability = float(proba)

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability)


if __name__=="__main__":
    app.run(host='0.0.0.0' , port=8080)