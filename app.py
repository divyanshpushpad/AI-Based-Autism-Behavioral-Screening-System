from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

# Initialize Flask app
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Load trained model
model = joblib.load('model/autism_model.pkl')

# Database Table
class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    age = db.Column(db.Integer)

    prediction = db.Column(db.String(100))

    date = db.Column(db.String(100))

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:

        features = [
            int(request.form['A1']),
            int(request.form['A2']),
            int(request.form['A3']),
            int(request.form['A4']),
            int(request.form['A5']),
            int(request.form['A6']),
            int(request.form['A7']),
            int(request.form['A8']),
            int(request.form['A9']),
            int(request.form['A10']),
            int(request.form['age'])
        ]

        input_data = pd.DataFrame([features], columns=[
            'A1_Score',
            'A2_Score',
            'A3_Score',
            'A4_Score',
            'A5_Score',
            'A6_Score',
            'A7_Score',
            'A8_Score',
            'A9_Score',
            'A10_Score',
            'age'
        ])

        prediction = model.predict(input_data)

        # Prediction Result
        if prediction[0] == 'YES':
            result = "Autism Traits Detected"
        else:
            result = "No Autism Traits Detected"

        # Save to Database
        new_prediction = Prediction(
            age=int(request.form['age']),
            prediction=result,
            date=datetime.now().strftime("%d %b %Y, %I:%M %p")
        )

        db.session.add(new_prediction)
        db.session.commit()

        return render_template(
            'result.html',
            prediction=result
        )

    except Exception as e:
        return str(e)

# History Route
@app.route('/history')
def history():

    records = Prediction.query.all()

    return render_template(
        'history.html',
        records=records
    )

# Dashboard Route
@app.route('/dashboard')
def dashboard():

    # Total records
    total_predictions = Prediction.query.count()

    # Autism detected count
    autism_detected = Prediction.query.filter_by(
        prediction="Autism Traits Detected"
    ).count()

    # No autism count
    no_autism = Prediction.query.filter_by(
        prediction="No Autism Traits Detected"
    ).count()

    # Create Pie Chart
    labels = ['Autism Detected', 'No Autism']
    values = [autism_detected, no_autism]

    plt.figure(figsize=(6,6))
    plt.pie(
    values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'width':0.5}
     )

    # Save chart
    chart_path = 'static/images/chart.png'

    plt.savefig(chart_path)
    plt.close()

    # Recent records
    recent_records = Prediction.query.order_by(
    Prediction.id.desc()
    ).limit(5).all()

    return render_template(
        'dashboard.html',
        total_predictions=total_predictions,
        autism_detected=autism_detected,
        no_autism=no_autism,
        recent_records=recent_records
    )

# Create Database
with app.app_context():
    db.create_all()

# Run App
if __name__ == '__main__':
    app.run(debug=True)