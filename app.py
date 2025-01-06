import boto3
import os
import pickle  # Replacing joblib with pickle
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from io import StringIO
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load data from S3 (this assumes that your dataset is in CSV format)
def load_data_from_s3(bucket_name, file_key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    # Read the CSV file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    # Load into a pandas DataFrame
    return pd.read_csv(StringIO(csv_content))

# Preprocess the data
def preprocess_data(df):
    df["date_ech"] = pd.to_datetime(df["date_ech"])  # Convert date to datetime format
    df["year"] = df["date_ech"].dt.year
    df["month"] = df["date_ech"].dt.month
    df["day"] = df["date_ech"].dt.day
    df["hour"] = df["date_ech"].dt.hour
    df["day_of_week"] = df["date_ech"].dt.dayofweek  # 0=Monday, 6=Sunday

    features = ['year', 'month', 'day', 'hour', 'day_of_week', 'x_wgs84', 'y_wgs84']
    target = 'lib_qual'  # Target is the air quality label (classification)

    encoder = LabelEncoder()
    df[target] = encoder.fit_transform(df[target])  # Convert to numeric values

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to download the model from S3
def download_model_from_s3(model_key, bucket_name):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    # Get the model object as bytes from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_data = response['Body'].read()  # Model data as bytes

    # Deserialize the model from bytes using pickle
    model = pickle.loads(model_data)  # Deserialize to get the model object
    return model


# Function to preprocess input data for prediction
def preprocess_input(input_data):
    """
    Preprocesses the input data in the same way as the model's training data.
    """
    # Convert the input data dictionary to a DataFrame
    df_input = pd.DataFrame([input_data])

    # Example of preprocessing: Date-related feature extraction (if needed)
    df_input["date_ech"] = pd.to_datetime(df_input["year"].astype(str) + "-" + df_input["month"].astype(str) + "-" + df_input["day"].astype(str))
    df_input["year"] = df_input["date_ech"].dt.year
    df_input["month"] = df_input["date_ech"].dt.month
    df_input["day"] = df_input["date_ech"].dt.day
    df_input["hour"] = df_input["date_ech"].dt.hour
    df_input["day_of_week"] = df_input["date_ech"].dt.dayofweek

    # Features used in model
    features = ['year', 'month', 'day', 'hour', 'day_of_week', 'x_wgs84', 'y_wgs84']
    df_input = df_input[features]  # Keep only the relevant features

    return df_input

# Function to map the numeric predictions back to original labels
def map_predictions_to_labels(predict_value):
    """
    Maps numeric predictions back to their original string labels using the LabelEncoder.
    """
    label_mapping = {
        0: 'Good',
        1: 'Moderate',
        2: 'Unhealthy for Sensitive Groups',
        3: 'Unhealthy',
        4: 'Very Unhealthy',
        5: 'Hazardous'
    }
    
    return label_mapping[predict_value]

# Function to make predictions using the trained model
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Main function to execute the experiment
def main(input_data):
    # Paths and model information
    bucket_name = os.getenv('BUCKET_NAME')
    model_key = os.getenv('PREDICTION_MODEL')

    # Download the model from S3
    model = download_model_from_s3(model_key, bucket_name)

    # Preprocess the input data
    processed_input = preprocess_input(input_data)

    # Make predictions
    predictions = make_predictions(model, processed_input)

    # Print predictions
    print("Predictions:", predictions)
    return predictions





def upload_raw_data_s3(bucket_name, file_path, new_data):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    # Step 1: Download the existing file
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        existing_data = response['Body'].read().decode('utf-8')
        existing_df = pd.read_csv(StringIO(existing_data))
    except s3_client.exceptions.NoSuchKey:
        # If the file doesn't exist, create a new DataFrame
        print(f"File {file_key} not found in bucket {bucket_name}. Creating a new file.")
        existing_df = pd.DataFrame()

    # Step 2: Create a DataFrame for the new data, skipping the header in the append
    new_data_df = pd.read_csv(StringIO(new_data))

    # Step 3: Append the new data to the existing DataFrame
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

    # Step 4: Upload the updated file back to S3
    csv_buffer = StringIO()
    updated_df.to_csv(csv_buffer, index=False)  # Ensure the header is written only once
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
    
    print(f"File {file_key} updated successfully in bucket {bucket_name}.")

# Example usage
bucket_name = os.getenv('BUCKET_NAME')
file_path = 'transfer/air_quality_data/Air_Quality_Occitanie_Update.csv'
new_data = """
date_ech,code_qual,lib_qual,coul_qual,date_dif,source,type_zone,code_zone,lib_zone,code_no2,code_so2,code_o3,code_pm10,code_pm25,x_wgs84,y_wgs84,x_reg,y_reg,epsg_reg,ObjectId,x,y
12/6/2024 12:00:00 AM,2,Moyen,#50CCAA,12/5/2024 9:00:00 AM,Atmo-Occitanie,EPCI,200066223,CC Arize Lèze,1,1,2,1,1,1.41493748413552,43.1798303556942,571047,6232472,2154,1,571047.014532619,6232472.20325349
12/6/2024 12:00:00 AM,2,Moyen,#50CCAA,12/5/2024 9:00:00 AM,Atmo-Occitanie,EPCI,200040905,CC Carmausin-Ségala,1,1,2,1,1,2.16642592656936,44.0562870012488,633209,6328940,2154,2,633209.459138345,6328939.757962
"""



# Flask route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json
        date_input = data.get('date_input')
        x = data.get('x_wgs84')
        y = data.get('y_wgs84')
        
        # Parse the date input
        date_obj = datetime.fromisoformat(date_input)  # Converts ISO 8601 format to a datetime object

        # Extract components from the date
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        hour = date_obj.hour
        day_of_week = date_obj.weekday()  # Returns 0 for Monday, 6 for Sunday
        
        # Example input data (user input or custom data)
        input_data = {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "day_of_week": day_of_week,  # For example, 3 means Thursday
            "x_wgs84": x,
            "y_wgs84": y
        }
        
        result = main(input_data)
        
        # Prepare the response
        response = {
            "date": date_input,
            "coordinates": {
                "x_wgs84": x,
                "y_wgs84": y
            },
            "category": map_predictions_to_labels(result[0])
        }

        logger.info(f"Prediction result: {result}")
        
        upload_raw_data_s3(bucket_name, file_path, new_data)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=7860)
    

