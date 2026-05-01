from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ============================================================
# DATABASE SETUP
# ============================================================

def init_database():
    """Create database and table if they don't exist"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age REAL,
            heart_rate REAL,
            systolic_bp REAL,
            oxygen REAL,
            temperature REAL,
            pain_level INTEGER,
            chronic_diseases INTEGER,
            arrival_mode INTEGER,
            arrival_mode_text TEXT,
            risk_level INTEGER,
            risk_name TEXT,
            confidence REAL,
            prob_low REAL,
            prob_medium REAL,
            prob_high REAL,
            prob_critical REAL,
            recommendation TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized: predictions.db")

# Call this when app starts
init_database()

def save_prediction_to_db(data, result):
    """Save prediction to database"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # Map arrival mode number to text
    arrival_modes = {0: "Walk-in", 1: "Wheelchair", 2: "Ambulance"}
    arrival_text = arrival_modes.get(data['arrival_mode'], "Unknown")
    
    # Risk names
    risk_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, age, heart_rate, systolic_bp, oxygen,
            temperature, pain_level, chronic_diseases, arrival_mode,
            arrival_mode_text, risk_level, risk_name, confidence,
            prob_low, prob_medium, prob_high, prob_critical, recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        data['age'],
        data['heart_rate'],
        data['systolic_bp'],
        data['oxygen'],
        data['temperature'],
        data['pain_level'],
        data['chronic_diseases'],
        data['arrival_mode'],
        arrival_text,
        result['risk_level'],
        risk_names[result['risk_level']],
        result['confidence'],
        result['probabilities']['low'],
        result['probabilities']['medium'],
        result['probabilities']['high'],
        result['probabilities']['critical'],
        result['recommendation_en']
    ))
    
    conn.commit()
    conn.close()
    print(f"✅ Prediction saved to database (ID: {cursor.lastrowid})")

def get_prediction_history(limit=50):
    """Get last predictions from database"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, age, heart_rate, systolic_bp, oxygen,
               temperature, pain_level, chronic_diseases, arrival_mode_text,
               risk_level, risk_name, confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'timestamp': row[1],
            'age': row[2],
            'heart_rate': row[3],
            'systolic_bp': row[4],
            'oxygen': row[5],
            'temperature': row[6],
            'pain_level': row[7],
            'chronic_diseases': row[8],
            'arrival_mode': row[9],
            'risk_level': row[10],
            'risk_name': row[11],
            'confidence': row[12]
        })
    
    return history

# ============================================================
# LOAD AI MODEL
# ============================================================

# Load model only (no SHAP explainer)
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['age', 'heart_rate', 'systolic_blood_pressure', 'oxygen_saturation',
            'body_temperature', 'pain_level', 'chronic_disease_count', 'arrival_mode']

risk_names = {
    0: {"en": "LOW RISK", "ar": "خطر منخفض", "fr": "RISQUES FAIBLES"},
    1: {"en": "MEDIUM RISK", "ar": "خطر متوسط", "fr": "RISQUE MOYEN"},
    2: {"en": "HIGH RISK", "ar": "خطر مرتفع", "fr": "RISQUE ÉLEVÉ"},
    3: {"en": "CRITICAL RISK", "ar": "خطر حرج", "fr": "RISQUE CRITIQUE"}
}

colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c", 3: "#8e44ad"}

actions = {
    0: {"en": "Normal waiting room. Reassess if symptoms change.",
        "ar": "غرفة انتظار عادية. أعد التقييم إذا تغيرت الأعراض.",
        "fr": "Salle d'attente normale. Réévaluer si les symptômes changent."},
    1: {"en": "Close monitoring. Check vital signs every 30 minutes.",
        "ar": "مراقبة دقيقة. فحص العلامات الحيوية كل 30 دقيقة.",
        "fr": "Surveillance étroite. Vérifier les signes vitaux toutes les 30 minutes."},
    2: {"en": "Admit to ICU immediately. Prepare for intervention.",
        "ar": "أدخل إلى العناية المركزة فورًا. استعد للتدخل.",
        "fr": "Admettre aux soins intensifs immédiatement. Préparer l'intervention."},
    3: {"en": "CODE BLUE! Immediate resuscitation required!",
        "ar": "كود أزرق! إنعاش فوري مطلوب!",
        "fr": "CODE BLEU! Réanimation immédiate requise!"}
}

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        patient_data = pd.DataFrame([[
            float(data['age']),
            float(data['heart_rate']),
            float(data['systolic_bp']),
            float(data['oxygen']),
            float(data['temperature']),
            int(data['pain_level']),
            int(data['chronic_diseases']),
            int(data['arrival_mode'])
        ]], columns=features)
        
        risk_level = int(model.predict(patient_data)[0])
        probabilities = model.predict_proba(patient_data)[0].tolist()
        confidence = max(probabilities) * 100
        
        result = {
            'success': True,
            'risk_level': risk_level,
            'risk_name_en': risk_names[risk_level]["en"],
            'risk_name_ar': risk_names[risk_level]["ar"],
            'risk_name_fr': risk_names[risk_level]["fr"],
            'color': colors[risk_level],
            'confidence': round(confidence, 1),
            'probabilities': {
                'low': round(probabilities[0] * 100, 1),
                'medium': round(probabilities[1] * 100, 1),
                'high': round(probabilities[2] * 100, 1),
                'critical': round(probabilities[3] * 100, 1)
            },
            'recommendation_en': actions[risk_level]["en"],
            'recommendation_ar': actions[risk_level]["ar"],
            'recommendation_fr': actions[risk_level]["fr"],
            'feature_impact': []
        }
        
        # Save to database
        save_prediction_to_db(data, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    """API endpoint to get prediction history"""
    history = get_prediction_history(50)
    return jsonify({'success': True, 'history': history})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all prediction history"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'History cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)