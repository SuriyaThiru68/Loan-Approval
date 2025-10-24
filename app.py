from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
 
logistic_model = joblib.load("model/logistic_model.pkl")
xgboost_model = joblib.load("model/xgboost_model.pkl")
scaler = joblib.load("model/scaler.pkl")  
map_gender = {"Male": 1, "Female": 0}
map_married = {"Yes": 1, "No": 0}
map_education = {"Graduate": 1, "Not Graduate": 0}
map_self_employed = {"Yes": 1, "No": 0}
map_credit = {"Good": 1, "Bad": 0}
map_property = {"Rural": 0, "Semiurban": 1, "Urban": 2}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form 
        model_type = form.get("model_type", "ensemble").lower() 
        Gender = map_gender[form["Gender"]]
        Married = map_married[form["Married"]]
        Dependents = int(form["Dependents"])
        Education = map_education[form["Education"]]
        Self_Employed = map_self_employed[form["Self_Employed"]]
        ApplicantIncome = float(form["ApplicantIncome"])
        CoapplicantIncome = float(form["CoapplicantIncome"])
        LoanAmount = float(form["LoanAmount"])
        Loan_Amount_Term = float(form["Loan_Amount_Term"])
        Credit_History = map_credit[form["Credit_History"]]
        Property_Area = map_property[form["Property_Area"]] 
        features = np.array([[
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area
        ]], dtype=float) 
        if model_type == "logistic":
            features_scaled = scaler.transform(features)
            proba = float(logistic_model.predict_proba(features_scaled)[0, 1])
            pred = int(proba >= 0.5)

        elif model_type == "xgboost":
            proba = float(xgboost_model.predict_proba(features)[0, 1])
            pred = int(proba >= 0.5)

        else:   
            features_scaled = scaler.transform(features)
            p_log = float(logistic_model.predict_proba(features_scaled)[0, 1])
            p_xgb = float(xgboost_model.predict_proba(features)[0, 1])
            w = 0.5  
            proba = w * p_log + (1 - w) * p_xgb
            pred = int(proba >= 0.5)

        label = "Approved" if pred == 1 else "Rejected"
        return jsonify({"prediction": pred, "label": label, "probability": round(proba, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
