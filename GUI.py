import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Step 1: Load and Train the Model

@st.cache_data
def load_and_train_model():
    df = pd.read_csv("laptop_price.csv", encoding='ISO-8859-1')

    # Clean and preprocess
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df['Memory'] = df['Memory'].str.replace('GB', '')
    df['Memory'] = df['Memory'].str.replace('TB', '000')
    df['Memory'] = df['Memory'].str.replace('+', ' ')
    df['Memory'] = df['Memory'].fillna('0')
    df['Memory'] = df['Memory'].apply(lambda x: sum([int(i) for i in x.split() if i.isdigit()]))

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

    def modify_cpu(x):
        if 'Intel Core i7' in x:
            return 'Intel Core i7'
        elif 'Intel Core i5' in x:
            return 'Intel Core i5'
        elif 'Intel Core i3' in x:
            return 'Intel Core i3'
        elif 'Intel' in x:
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
    
    df['Cpu Brand'] = df['Cpu'].apply(modify_cpu)
    df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Other')

    df['HDD'] = 0
    df['SSD'] = df['Memory']
    df['Hybrid'] = 0
    df['Flash_Storage'] = 0

    df.drop(columns=['ScreenResolution', 'Cpu', 'Gpu', 'Memory', 'laptop_ID', 'Product'], errors='ignore', inplace=True)

    x= df.drop('Price_euros', axis=1)
    y = df['Price_euros']

    categorical_cols = x.select_dtypes(include='object').columns.tolist()
    numerical_cols = x.select_dtypes(exclude='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline.fit(x_train, y_train)

    return pipeline

#  Call the model loading function 
model = load_and_train_model()


# Step 2: Streamlit GUI

st.title("💻 Laptop Price Predictor")

company = st.selectbox("Company", ['Dell', 'HP', 'Apple', 'Lenovo', 'Acer', 'Asus', 'MSI', 'Toshiba', 'Samsung', 'LG', 'Other'])
laptop_type = st.selectbox("Type", ['Ultrabook', 'Gaming', 'Notebook', 'Netbook', 'Workstation'])
inches = st.slider("Screen Size (Inches)", 10.0, 20.0, 15.6)
resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 'Touchscreen IPS', 'Touchscreen'])
cpu = st.selectbox("CPU", ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'AMD Ryzen', 'Other'])
ram = st.slider("RAM (GB)", 2, 64, 8)
memory = st.text_input("Storage (e.g. 256GB, 512GB + 1TB)", "512GB + 1TB")
gpu = st.selectbox("GPU", ['Intel HD', 'Nvidia GTX', 'AMD Radeon', 'Other'])
os = st.selectbox("Operating System", ['Windows', 'Mac', 'Linux', 'Chrome OS', 'No OS'])
weight = st.slider("Weight (kg)", 0.5, 3.0, 1.5)

def convert_input_memory(mem):
    mem = mem.replace('GB', '').replace('TB', '000')
    mem = mem.replace('+', ' ')
    parts = mem.split()
    return sum([int(part) for part in parts if part.isdigit()])

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['Ram'] = df['Ram'].astype(int)
    df['Weight'] = df['Weight'].astype(float)

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    df.drop('ScreenResolution', axis=1, inplace=True)

    def modifying(cpu):
        if 'Intel Core i7' in cpu:
            return 'Intel Core i7'
        elif 'Intel Core i5' in cpu:
            return 'Intel Core i5'
        elif 'Intel Core i3' in cpu:
            return 'Intel Core i3'
        elif 'Intel' in cpu:
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

    df['Cpu Brand'] = df['Cpu'].apply(modifying)
    df.drop('Cpu', axis=1, inplace=True)

    df['HDD'] = 0
    df['SSD'] = df['Memory']
    df['Hybrid'] = 0
    df['Flash_Storage'] = 0
    df.drop('Memory', axis=1, inplace=True)

    df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Other')
    df.drop('Gpu', axis=1, inplace=True)

    return df


# Step 3: Prediction

if st.button("Predict Price"):
    input_data = {
        'Company': company,
        'TypeName': laptop_type,
        'Inches': inches,
        'ScreenResolution': resolution,
        'Cpu': cpu,
        'Ram': ram,
        'Memory': convert_input_memory(memory),
        'Gpu': gpu,
        'OpSys': os,
        'Weight': weight
    }

    input_df = preprocess_input(input_data)

    predicted_price = model.predict(input_df)[0]

    st.success(f"Estimated Laptop Price: €{predicted_price:.2f}")
