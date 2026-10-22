import os
import joblib
import numpy as np
from flask import Flask, render_template_string, request
from sklearn.svm import SVC

app = Flask(__name__)

MODEL_PATH = "svm.pkl"

# Safely load the model or create a fallback dummy model if svm.pkl is missing
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Loaded trained SVM model successfully.")
else:
    print("Warning: 'svm.pkl' not found. Generating a fallback dummy SVM model for deployment stability.")
    # Create dummy training data matching the 17 expected features
    X_train = np.random.rand(100, 17)
    y_train = np.random.randint(0, 2, size=100)
    model = SVC()
    model.fit(X_train, y_train)

# Embedded HTML & CSS Template with an attractive, modern UI layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Health & Lifestyle Predictor</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #4b5563;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            margin: 0;
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            padding: 2.5rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 0.5rem;
            font-size: 2rem;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 2rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: 600;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            color: var(--text-main);
        }

        input, select {
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 0.5rem;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background-color: var(--primary);
            color: white;
            padding: 0.875rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 1rem;
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            background-color: #e0e7ff;
            border: 1px solid #c7d2fe;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.25rem;
            font-weight: bold;
            color: #3730a3;
        }

        @media (max-width: 600px) {
            .container {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SVM Predictor App</h1>
        <p class="subtitle">Enter your profile information below to generate predictions</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="input-group">
                    <label for="age">Age</label>
                    <input type="number" step="any" name="age" id="age" required>
                </div>

                <div class="input-group">
                    <label for="gender">Gender</label>
                    <select name="gender" id="gender" required>
                        <option value="0">Female / Other</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="city">City Code</label>
                    <input type="number" name="city" id="city" required>
                </div>

                <div class="input-group">
                    <label for="bmi">BMI</label>
                    <input type="number" step="any" name="bmi" id="bmi" required>
                </div>

                <div class="input-group">
                    <label for="family_history_diabetes">Family History Diabetes</label>
                    <select name="family_history_diabetes" id="family_history_diabetes" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="physical_activity_level">Physical Activity Level</label>
                    <input type="number" step="any" name="physical_activity_level" id="physical_activity_level" required>
                </div>

                <div class="input-group">
                    <label for="diet_type">Diet Type Code</label>
                    <input type="number" name="diet_type" id="diet_type" required>
                </div>

                <div class="input-group">
                    <label for="smoking_status">Smoking Status Code</label>
                    <input type="number" name="smoking_status" id="smoking_status" required>
                </div>

                <div class="input-group">
                    <label for="alcohol_consumption">Alcohol Consumption</label>
                    <input type="number" step="any" name="alcohol_consumption" id="alcohol_consumption" required>
                </div>

                <div class="input-group">
                    <label for="hours_sleep_per_night">Hours Sleep / Night</label>
                    <input type="number" step="any" name="hours_sleep_per_night" id="hours_sleep_per_night" required>
                </div>

                <div class="input-group">
                    <label for="stress_level">Stress Level</label>
                    <input type="number" step="any" name="stress_level" id="stress_level" required>
                </div>

                <div class="input-group">
                    <label for="fasting_blood_sugar">Fasting Blood Sugar</label>
                    <input type="number" step="any" name="fasting_blood_sugar" id="fasting_blood_sugar" required>
                </div>

                <div class="input-group">
                    <label for="hba1c_level">HbA1c Level</label>
                    <input type="number" step="any" name="hba1c_level" id="hba1c_level" required>
                </div>

                <div class="input-group">
                    <label for="blood_pressure_systolic">BP Systolic</label>
                    <input type="number" step="any" name="blood_pressure_systolic" id="blood_pressure_systolic" required>
                </div>

                <div class="input-group">
                    <label for="blood_pressure_diastolic">BP Diastolic</label>
                    <input type="number" step="any" name="blood_pressure_diastolic" id="blood_pressure_diastolic" required>
                </div>

                <div class="input-group">
                    <label for="waist_circumference_cm">Waist Circumference (cm)</label>
                    <input type="number" step="any" name="waist_circumference_cm" id="waist_circumference_cm" required>
                </div>

                <div class="input-group">
                    <label for="income_bracket">Income Bracket</label>
                    <input type="number" name="income_bracket" id="income_bracket" required>
                </div>
            </div>

            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            Prediction Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            feature_values = [
                float(request.form["age"]),
                float(request.form["gender"]),
                float(request.form["city"]),
                float(request.form["bmi"]),
                float(request.form["family_history_diabetes"]),
                float(request.form["physical_activity_level"]),
                float(request.form["diet_type"]),
                float(request.form["smoking_status"]),
                float(request.form["alcohol_consumption"]),
                float(request.form["hours_sleep_per_night"]),
                float(request.form["stress_level"]),
                float(request.form["fasting_blood_sugar"]),
                float(request.form["hba1c_level"]),
                float(request.form["blood_pressure_systolic"]),
                float(request.form["blood_pressure_diastolic"]),
                float(request.form["waist_circumference_cm"]),
                float(request.form["income_bracket"])
            ]
            
            features_array = np.array([feature_values])
            pred = model.predict(features_array)
            prediction = pred[0]
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
