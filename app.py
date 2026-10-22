import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML + Tailwind CSS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-slate-100 min-h-screen py-10 px-4">
    <div class="max-w-4xl mx-auto bg-slate-800/80 backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden border border-slate-700">
        
        <!-- Header -->
        <div class="bg-gradient-to-r from-blue-600 to-indigo-600 px-8 py-6 text-center">
            <h1 class="text-3xl font-extrabold tracking-tight text-white">SVM Risk Assessment Portal</h1>
            <p class="text-blue-100 mt-1 text-sm">Enter the patient parameters below to generate an instant prediction.</p>
        </div>

        <!-- Form / Result Section -->
        <div class="p-8">
            {% if prediction %}
            <div class="mb-8 p-6 rounded-xl text-center border {% if prediction == 'High' %}bg-red-950/50 border-red-500 text-red-200{% elif prediction == 'Moderate' %}bg-amber-950/50 border-amber-500 text-amber-200{% else %}bg-emerald-950/50 border-emerald-500 text-emerald-200{% endif %}">
                <h3 class="text-lg font-medium uppercase tracking-wider">Prediction Result</h3>
                <p class="text-4xl font-black mt-2">{{ prediction }} Risk Level</p>
                <a href="/" class="inline-block mt-4 px-6 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white text-sm font-semibold transition">Make Another Prediction</a>
            </div>
            {% endif %}

            <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- Age -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Age</label>
                    <input type="number" name="age" required value="{{ form_data.get('age', 35) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Gender -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Gender</label>
                    <select name="gender" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="0">Female / Other</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <!-- City -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">City Code / Category</label>
                    <input type="number" name="city" required value="{{ form_data.get('city', 1) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- BMI -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">BMI</label>
                    <input type="step" step="0.1" name="bmi" required value="{{ form_data.get('bmi', 24.5) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Family History Diabetes -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Family History of Diabetes</label>
                    <select name="family_history_diabetes" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <!-- Physical Activity Level -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Physical Activity Level (0-2)</label>
                    <input type="number" name="physical_activity_level" required value="{{ form_data.get('physical_activity_level', 1) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Diet Type -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Diet Type (0-2)</label>
                    <input type="number" name="diet_type" required value="{{ form_data.get('diet_type', 1) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Smoking Status -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Smoking Status (0-2)</label>
                    <input type="number" name="smoking_status" required value="{{ form_data.get('smoking_status', 0) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Alcohol Consumption -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Alcohol Consumption (0-2)</label>
                    <input type="number" name="alcohol_consumption" required value="{{ form_data.get('alcohol_consumption', 0) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Hours Sleep Per Night -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Hours Sleep Per Night</label>
                    <input type="number" step="0.1" name="hours_sleep_per_night" required value="{{ form_data.get('hours_sleep_per_night', 7) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Stress Level -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Stress Level (1-10)</label>
                    <input type="number" name="stress_level" required value="{{ form_data.get('stress_level', 5) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Fasting Blood Sugar -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Fasting Blood Sugar</label>
                    <input type="number" step="0.1" name="fasting_blood_sugar" required value="{{ form_data.get('fasting_blood_sugar', 95) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- HbA1c Level -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">HbA1c Level</label>
                    <input type="number" step="0.1" name="hba1c_level" required value="{{ form_data.get('hba1c_level', 5.5) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Blood Pressure Systolic -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Blood Pressure Systolic</label>
                    <input type="number" name="blood_pressure_systolic" required value="{{ form_data.get('blood_pressure_systolic', 120) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Blood Pressure Diastolic -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Blood Pressure Diastolic</label>
                    <input type="number" name="blood_pressure_diastolic" required value="{{ form_data.get('blood_pressure_diastolic', 80) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Waist Circumference CM -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Waist Circumference (cm)</label>
                    <input type="number" step="0.1" name="waist_circumference_cm" required value="{{ form_data.get('waist_circumference_cm', 85) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Income Bracket -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-1">Income Bracket (0-3)</label>
                    <input type="number" name="income_bracket" required value="{{ form_data.get('income_bracket', 1) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Submit Button -->
                <div class="md:col-span-2 mt-4">
                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition duration-200">
                        Run Prediction
                    </button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the exact order trained by your model
        form_data = request.form
        features = [
            float(form_data['age']),
            float(form_data['gender']),
            float(form_data['city']),
            float(form_data['bmi']),
            float(form_data['family_history_diabetes']),
            float(form_data['physical_activity_level']),
            float(form_data['diet_type']),
            float(form_data['smoking_status']),
            float(form_data['alcohol_consumption']),
            float(form_data['hours_sleep_per_night']),
            float(form_data['stress_level']),
            float(form_data['fasting_blood_sugar']),
            float(form_data['hba1c_level']),
            float(form_data['blood_pressure_systolic']),
            float(form_data['blood_pressure_diastolic']),
            float(form_data['waist_circumference_cm']),
            float(form_data['income_bracket'])
        ]
        
        # Reshape for sklearn model prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=prediction, form_data=form_data)
    except Exception as e:
        return f"<h3>Error during prediction:</h3><p>{e}</p><p><a href='/'>Go Back</a></p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
