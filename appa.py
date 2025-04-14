from flask import Flask, render_template, request, url_for, json, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from joblib import dump, load
model_path = 'kmeans_modelG.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
dump(model, 'kmeans_joblibG.joblib')
# model = load('kmeans_joblibG.joblib')
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# from semester_Project import preprocess_data

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='')
# app = Flask(__name__)



model = pickle.load(open('kmeans_modelG.pkl', 'rb'))
template_dir = os.path.abspath('templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reloading
app.config['STATIC_FOLDER'] = os.path.abspath('static')



def load_and_clean_data(file_path):
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    retail['StayingRate'] = 1 - retail['Exit_Rate']

    rfm_St = retail.groupby('CustomerID') ['StayingRate'].sum().reset_index()
    rfm_St.columns = ['CustomerID', 'StayingRate']
    retail['Frequency'] = retail['Total_Page_Views'] 
    rfm_Fr = retail.groupby('CustomerID') ['Frequency'].sum().reset_index()
    rfm_Fr.columns = ['CustomerID', 'Frequency']
    
    
    retail['Length'] = retail['Month']
    
    rfm_Le = retail.groupby('CustomerID') ['Length'].sum().reset_index()
    rfm_Le.columns = ['CustomerID', 'Length']
    retail['Month'] = pd.to_numeric(retail['Month'], errors='coerce') 
    retail['Recency']=12-row['Month'] + 1
    rfm_De = retail.groupby('CustomerID') ['Recency'].sum().reset_index()
    rfm_De.columns = ['CustomerID', 'Recency']
    


    
    rfm_Le['CustomerID'] = rfm_Le['CustomerID'].astype(str)

    rfm_Fr['CustomerID'] = rfm_Fr['CustomerID'].astype(str)
    rfm_St['CustomerID'] = rfm_St['CustomerID'].astype(str)
    rfm_De['CustomerID'] = rfm_De['CustomerID'].astype(str)
    


    rfm = pd.merge(rfm_Le, rfm_Fr, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_St, on='CustomerID', how='inner')
    rfm= pd.merge(rfm, rfm_De, on='CustomerID', how='inner')
    
   
    



    rfm.columns = ['CustomerID', 'StayingRate', 'Frequency', 'Length', 'Recency']
   

    # Convert 'Amount', 'Frequency', and 'Recency' to numeric with error handling
    numeric_columns = ['StayingRate', 'Frequency','Length', 'Recency']
    try:
        rfm[numeric_columns] = rfm[numeric_columns].apply(pd.to_numeric, errors='coerce')
    except pd.errors.OverflowError as e:
        # Handle overflow errors if necessary
        print(f"Error converting columns to numeric: {e}")

    # Drop rows with NaN values in any of the numeric columns
    rfm = rfm.dropna(subset=numeric_columns)

    # Make sure the conversion was successful for all columns
    if rfm[numeric_columns].dtypes.all() != 'object':
        print("All columns successfully converted to numeric.")

    # Remove outliers using IQR method
    for column in numeric_columns:
        Q1 = rfm[column].quantile(0.05)
        Q3 = rfm[column].quantile(0.95)
        IQR = Q3 - Q1

        rfm = rfm[(rfm[column] >= Q1 - 1.5 * IQR) & (rfm[column] <= Q3 + 1.5 * IQR)]

    return rfm



def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['StayingRate', 'Frequency','Length', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame (rfm_df_scaled)
    rfm_df_scaled.columns = ['StayingRate', 'Frequency','Length','Recency']
    return rfm,rfm_df_scaled;



@app.route('/')
def home():
    template_path = os.path.join(app.root_path, app.template_folder, 'index.html')
    print(f"Template Path: {template_path}")
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os. path.join(os.getcwd(), file.filename)
    # rfm=load_and_clean_data(file_path)
    file.save(file_path)
    df = preprocess_data(file_path) [1] 
    results_df = model.predict(df) 
    df_with_id = preprocess_data(file_path) [0]
    df_with_id['Cluster_Id'] = results_df
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
     #
    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id)
    recen_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recen_img_path)
    plt.close()  # Close the figure
    try:
        # ... (your existing code for generating images)

        response = {
            'stayingrate_img': stayingrate_img_path,
            'freq_img' :  freq_img_path,
            'length_img': length_img_path,
            'recency_img':recen_img_path
               
    
            
        }

        return render_template('result.html', **response)
    except Exception as e:
        return json.dumps({'error': str(e)})

@app.route('/get_images')
def get_images():
    images = {
        "stayingrate_img": url_for('static', filename='ClusterId_StayingRate.png'),
        "freq_img": url_for('static', filename='Cluster Id_Frequency.png'),
        "length_img": url_for('static', filename='Cluster Id_Length.png'),
        "recency_img": url_for('static', filename='ClusterId_Recency.png')
    }
    return images


if __name__=="__main__" :
    app.run(debug=True)



