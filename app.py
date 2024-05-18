from flask import Flask, request, jsonify 
import joblib 
import pandas as pd 
 
app = Flask(__name__) 
 
# Load model and scaler 
model = joblib.load('treebag_model.pkl') 
scaler = joblib.load('scaler.pkl') 
 
@app.route('/') 
def home(): 
    return "Welcome to the mortality prediction API!" 
 
@app.route('/predict', methods=['POST']) 
def predict(): 
    data = request.json 
    df = pd.DataFrame(data, index=[0]) 
    df_scaled = scaler.transform(df) 
    prediction = model.predict(df_scaled) 
    return jsonify({'prediction': int(prediction[0])}) 
 
if __name__ == '__main__': 
    app.run(debug=True) 
