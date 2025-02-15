from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import json
import requests

# Create base directory for the application
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / 'database'
DB_FILE = DB_DIR / 'users.db'

# Create database directory if it doesn't exist
DB_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Zerodha API credentials
ZERODHA_API_KEY = "your-zerodha-api-key"
ZERODHA_API_SECRET = "your-zerodha-api-secret"

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(100), nullable=True)
    name = db.Column(db.String(100))
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    zerodha_access_token = db.Column(db.String(100), nullable=True)
    searches = db.relationship('SearchHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    search_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    prediction_days = db.Column(db.Integer, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    # Pass google_enabled=False to template to hide Google login button
    return render_template('login.html', google_enabled=False)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'})

        user = User(email=email, name=name)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)})

    return render_template('register.html')

@app.route('/login/password', methods=['POST'])
def login_with_password():
    email = request.form.get('email')
    password = request.form.get('password')
    
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    
    return jsonify({'error': 'Invalid email or password'})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        stock_symbol = request.form['stock_symbol']
        prediction_days = int(request.form['prediction_days'])

        # Save search history
        search = SearchHistory(
            user_id=current_user.id,
            stock_symbol=stock_symbol,
            prediction_days=prediction_days
        )
        db.session.add(search)
        db.session.commit()

        # Fetch data
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="5y")
        
        if data.empty:
            return jsonify({'error': 'No data found for the given stock symbol'})

        # Process data
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        data.dropna(subset=['Close'], inplace=True)

        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[["Close"]])
        
        # Prepare training data
        X, y = prepare_data(scaled_data, 50)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(50, 1)),
            LSTM(50),
            Dense(25, activation="relu"),
            Dense(1)
        ])
        
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

        # Make predictions
        future_prices = predict_future(model, scaled_data, scaler, prediction_days)
        
        # Prepare response data
        historical_dates = data.index.strftime('%Y-%m-%d').tolist()
        historical_prices = data['Close'].tolist()
        future_dates = pd.date_range(start=data.index[-1], periods=prediction_days + 1)[1:].strftime('%Y-%m-%d').tolist()

        response_data = {
            'current_price': float(data['Close'].iloc[-1]),
            'current_volume': int(data['Volume'].iloc[-1]),
            'historical_dates': historical_dates,
            'historical_prices': historical_prices,
            'future_dates': future_dates,
            'predicted_prices': future_prices.flatten().tolist(),
            'price_change': float(future_prices[-1][0] - data['Close'].iloc[-1]),
            'percent_change': float((future_prices[-1][0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search-history')
@login_required
def search_history():
    searches = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.search_date.desc())\
        .limit(10)\
        .all()
    return jsonify([{
        'stock_symbol': s.stock_symbol,
        'prediction_days': s.prediction_days,
        'search_date': s.search_date.strftime('%Y-%m-%d %H:%M:%S')
    } for s in searches])

def prepare_data(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def predict_future(model, data, scaler, days):
    last_data = data[-50:].reshape(1, 50, 1)
    predictions = []
    
    for _ in range(days):
        pred = model.predict(last_data, verbose=0)[0, 0]
        predictions.append(pred)
        last_data = np.roll(last_data, -1)
        last_data[0, -1, 0] = pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

if __name__ == '__main__':
    with app.app_context():
        try:
            if not DB_FILE.exists():
                db.create_all()
                print(f"Database created successfully at {DB_FILE}")
            else:
                print(f"Database already exists at {DB_FILE}")
        except Exception as e:
            print(f"Error creating database: {e}")
    app.run(debug=True) 