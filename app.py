from flask import Flask, request, jsonify
import pandas as pd
import logging
from functions import model, cslib, etl, viz
import os

#Draft Version of an API with Train, Predict, and Logfile Endpoints

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='api.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables to store models and data
models = {}
data = {}

@app.route('/train', methods=['POST'])
def train():
    """
    Endpoint to train the model.
    """
    try:
        # Load data
        data_dir = request.json.get('data_dir')
        ts_files_dir = request.json.get('ts_files_dir')
        dfs = cslib.fetch_ts(data_dir, ts_files_dir, clean=True)
        
        # Preprocess data and train models
        for key, df in dfs.items():
            df_processed = model.preprocess_data(df)
            X, y = model.split_into_predictors_targets(df_processed)
            X_train, X_test, y_train, y_test = model.split_into_training_test(X, y)
            _, _, _, lr_model, rf_model, xgb_model = model.train_predict_evaluate(X_train, y_train, X_test, y_test)
            models[key] = {'lr': lr_model, 'rf': rf_model, 'xgb': xgb_model}
            data[key] = df_processed
        
        logging.info("Training completed successfully.")
        return jsonify({"message": "Training completed successfully."}), 200
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions.
    """
    try:
        # Get input data
        country = request.json.get('country', 'all')
        date = request.json.get('date')
        
        # Convert date to features
        month = pd.to_datetime(date).month
        day = pd.to_datetime(date).day
        test_date = {'current_month': month, 'current_day': day}
        new_test_df = pd.DataFrame(data=test_date, index=([0]))
        
        # Make predictions
        predictions = {}
        for model_name, model_dict in models.items():
            predictions[model_name] = {
                'lr': model_dict['lr'].predict(new_test_df).tolist(),
                'rf': model_dict['rf'].predict(new_test_df).tolist(),
                'xgb': model_dict['xgb'].predict(new_test_df).tolist()
            }
        
        logging.info(f"Prediction made for date: {date}")
        return jsonify(predictions), 200
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Endpoint to retrieve logs.
    """
    try:
        with open('api.log', 'r') as log_file:
            logs = log_file.read()
        return jsonify({"logs": logs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)