from flask import Flask, render_template, request, url_for, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump, load
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='')

# Load pre-trained model
model_path = 'kmeans_modelG.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
dump(model, 'kmeans_joblibG.joblib')  # Save model to joblib if necessary

# Load the model again using joblib (if needed)
# model = load('kmeans_joblibG.joblib')

# Data preprocessing functions
def load_and_clean_data(file_path):
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    # Calculate staying rate and frequency
    retail['StayingRate'] = 1 - retail['Exit_Rate']
    rfm_St = retail.groupby('CustomerID')['StayingRate'].sum().reset_index()
    rfm_St.columns = ['CustomerID', 'StayingRate']
    retail['Frequency'] = retail['Total_Page_Views']
    rfm_Fr = retail.groupby('CustomerID')['Frequency'].sum().reset_index()
    rfm_Fr.columns = ['CustomerID', 'Frequency']
    
    # Calculate length and recency
    retail['Length'] = retail['Month']
    rfm_Le = retail.groupby('CustomerID')['Length'].sum().reset_index()
    rfm_Le.columns = ['CustomerID', 'Length']
    retail['Month'] = pd.to_numeric(retail['Month'], errors='coerce')
    retail['Recency'] = 12 - retail['Month'] + 1
    rfm_De = retail.groupby('CustomerID')['Recency'].sum().reset_index()
    rfm_De.columns = ['CustomerID', 'Recency']
    
    # Merge data frames to create the final RFM dataset
    rfm_Le['CustomerID'] = rfm_Le['CustomerID'].astype(str)
    rfm_Fr['CustomerID'] = rfm_Fr['CustomerID'].astype(str)
    rfm_St['CustomerID'] = rfm_St['CustomerID'].astype(str)
    rfm_De['CustomerID'] = rfm_De['CustomerID'].astype(str)

    rfm = pd.merge(rfm_Le, rfm_Fr, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_St, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_De, on='CustomerID', how='inner')

    rfm.columns = ['CustomerID', 'StayingRate', 'Frequency', 'Length', 'Recency']
    
    # Convert columns to numeric and drop NaNs
    numeric_columns = ['StayingRate', 'Frequency', 'Length', 'Recency']
    rfm[numeric_columns] = rfm[numeric_columns].apply(pd.to_numeric, errors='coerce')
    rfm = rfm.dropna(subset=numeric_columns)

    # Remove outliers using IQR method
    for column in numeric_columns:
        Q1 = rfm[column].quantile(0.05)
        Q3 = rfm[column].quantile(0.95)
        IQR = Q3 - Q1
        rfm = rfm[(rfm[column] >= Q1 - 1.5 * IQR) & (rfm[column] <= Q3 + 1.5 * IQR)]

    return rfm


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['StayingRate', 'Frequency', 'Length', 'Recency']]

    # Impute missing values using SimpleImputer before scaling
    imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean
    rfm_df_imputed = imputer.fit_transform(rfm_df)

    # Scale the data
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df_imputed)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns=['StayingRate', 'Frequency', 'Length', 'Recency'])
    
    return rfm, rfm_df_scaled


# Routes
@app.route('/')
def home():
    template_path = os.path.join(app.root_path, app.template_folder, 'index.html')
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file and save it
        file = request.files['file']
        file_path = os.path.join(os.getcwd(), file.filename)
        file.save(file_path)

        # Preprocess the data
        df_scaled = preprocess_data(file_path)[1]

        # Predict the clusters using the pre-trained model
        results_df = model.predict(df_scaled)

        # Add the predicted cluster labels back to the original dataframe
        df_with_id = preprocess_data(file_path)[0]
        df_with_id['Cluster_Id'] = results_df

        # Create visualizations
        sns.stripplot(x='Cluster_Id', y='StayingRate', data=df_with_id)
        stayingrate_img_path = 'static/ClusterId_StayingRate.png'
        plt.savefig(stayingrate_img_path)
        plt.close()

        sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id)
        freq_img_path = 'static/ClusterId_Frequency.png'
        plt.savefig(freq_img_path)
        plt.close()

        sns.stripplot(x='Cluster_Id', y='Length', data=df_with_id)
        length_img_path = 'static/ClusterId_Length.png'
        plt.savefig(length_img_path)
        plt.close()

        sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id)
        recen_img_path = 'static/ClusterId_Recency.png'
        plt.savefig(recen_img_path)
        plt.close()

        # Return the image paths for the result page
        response = {
            'stayingrate_img': stayingrate_img_path,
            'freq_img': freq_img_path,
            'length_img': length_img_path,
            'recency_img': recen_img_path
        }
        return render_template('result.html', **response)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_images')
def get_images():
    images = {
        "stayingrate_img": url_for('static', filename='ClusterId_StayingRate.png'),
        "freq_img": url_for('static', filename='ClusterId_Frequency.png'),
        "length_img": url_for('static', filename='ClusterId_Length.png'),
        "recency_img": url_for('static', filename='ClusterId_Recency.png')
    }
    return jsonify(images)


if __name__ == "__main__":
    app.run(debug=True)
