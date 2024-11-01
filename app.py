from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('crop_pred.pkl', 'rb'))

# Load label encoders for STATE and CROP
state_encoder = LabelEncoder()
state_encoder.fit(pd.read_csv(r"C:\Users\ROHITH\Downloads\Telegram Desktop\indiancrop_dataset.csv")["STATE"])

le = LabelEncoder()
le.fit(pd.read_csv(r"C:\Users\ROHITH\Downloads\Telegram Desktop\indiancrop_dataset.csv")["CROP"])

# Route for crop.html (initial page)
@app.route('/')
def show_crop_page():
    return render_template('crop.html')

# Route for index.html
@app.route('/index')
def home():
    return render_template('index.html', states=state_encoder.classes_)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N_SOIL = float(request.form['N_SOIL'])
        P_SOIL = float(request.form['P_SOIL'])
        K_SOIL = float(request.form['K_SOIL'])
        TEMPERATURE = float(request.form['TEMPERATURE'])
        HUMIDITY = float(request.form['HUMIDITY'])
        ph = float(request.form['ph'])
        RAINFALL = float(request.form['RAINFALL'])
        STATE = request.form['STATE']
        CROP_PRICE = float(request.form['CROP_PRICE'])

        # Encode STATE
        STATE_encoded = state_encoder.transform([STATE])[0]

        # Prepare data for prediction
        data = np.array([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL, STATE_encoded, CROP_PRICE]])
        prediction = model.predict(data)
        predicted_crop = le.inverse_transform(prediction)[0]

        # Redirect to main.html with the prediction and image filename
        return redirect(url_for('main', prediction_text=f'The predicted crop type is: {predicted_crop}', image_filename=f'{predicted_crop.lower()}.jpg'))

# Route for main.html
@app.route('/main')
def main():
    prediction_text = request.args.get('prediction_text', '')
    image_filename = request.args.get('image_filename', '')
    return render_template('main.html', prediction_text=prediction_text, image_filename=image_filename)

if __name__ == "__main__":
    app.run(debug=True)
