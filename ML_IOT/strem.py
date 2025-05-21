import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('ms_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: ms_dataset.csv not found in the current directory!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def train_and_save_model():
    """Train the model and save it with all necessary data"""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Prepare features and target
        X = df.drop(['ID', 'MS_Diagnosis'], axis=1)
        y = df['MS_Diagnosis']
        
        # Handle categorical variables
        X = pd.get_dummies(X, columns=['Gender'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save all necessary data
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }
        
        try:
            with open('ms_model_data.pkl', 'wb') as file:
                pickle.dump(model_data, file)
            return model_data
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            st.stop()
            
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.stop()

def load_or_train_model():
    """Load existing model or train new one if needed"""
    try:
        if os.path.exists('ms_model_data.pkl') and os.path.getsize('ms_model_data.pkl') > 0:
            with open('ms_model_data.pkl', 'rb') as file:
                return pickle.load(file)
        else:
            st.info("Training new model...")
            model_data = train_and_save_model()
            st.success("Model trained successfully!")
            return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Training new model...")
        model_data = train_and_save_model()
        st.success("Model trained successfully!")
        return model_data

def create_input_dataframe(input_dict, feature_names):
    """Create properly formatted DataFrame from input values"""
    df = pd.DataFrame([input_dict])
    
    # Create gender dummy variables
    gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
    df = pd.concat([df.drop('Gender', axis=1), gender_dummies], axis=1)
    
    # Ensure all features from training are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
            
    # Reorder columns to match training data
    df = df[feature_names]
    
    return df

def main():
    st.set_page_config(page_title="MS Diagnosis Prediction", layout="wide")
    
    st.title("Multiple Sclerosis Diagnosis Prediction")
    st.markdown("""
    This application predicts Multiple Sclerosis diagnosis based on patient physiological and behavioral data.
    Please enter the required information below.
    """)
    
    # Load or train model at startup
    try:
        model_data = load_or_train_model()
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
    except Exception as e:
        st.error(f"Fatal error: Could not load or train model: {str(e)}")
        st.stop()
    
    # Create input form
    st.header("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input(
            "Age (years)", 
            min_value=20, max_value=60, value=30,
            help="Most common age range for MS diagnosis is 20-60 years"
        )
        
        gender = st.selectbox("Gender", ['F', 'M'])
        
        heart_rate = st.number_input(
            "Heart Rate (bpm)", 
            min_value=60.0, max_value=100.0, value=75.0, step=1.0,
            help="Normal range 60-100 bpm, MS patients tend towards higher values"
        )
        
        respiration_rate = st.number_input(
            "Respiration Rate (breaths/min)", 
            min_value=12.0, max_value=20.0, value=15.0, step=1.0,
            help="Typical adult respiration rate range"
        )
        
    with col2:
        skin_temp = st.number_input(
            "Skin Temperature (°C)", 
            min_value=33.0, max_value=36.0, value=34.5, step=0.1,
            help="Accounts for temperature regulation issues in MS"
        )
        
        blood_pulse = st.number_input(
            "Blood Pulse Wave (bpm)", 
            min_value=60.0, max_value=100.0, value=75.0, step=1.0,
            help="Represented by pulse rate"
        )
        
        hrv = st.number_input(
            "Heart Rate Variability (ms)", 
            min_value=20.0, max_value=70.0, value=45.0, step=1.0,
            help="SDNN values, lower values common in MS patients"
        )
        
        gsr = st.number_input(
            "Galvanic Skin Response (µS)", 
            min_value=1.0, max_value=10.0, value=5.0, step=0.1,
            help="Measured in microsiemens, lower values typical in MS"
        )

    with col3:
        energy = st.number_input(
            "Energy Expenditure (kcal/day)", 
            min_value=1500.0, max_value=2500.0, value=2000.0, step=50.0,
            help="Daily energy expenditure, may be reduced in MS patients"
        )
        
        steps = st.number_input(
            "Daily Steps", 
            min_value=3000, max_value=8000, value=5000, step=100,
            help="Daily step count, often lower in MS patients with mobility issues"
        )
        
        phone_locks = st.number_input(
            "Phone Lock/Unlock Time (seconds)", 
            min_value=1.5, max_value=5.0, value=2.5, step=0.1,
            help="Time taken to lock/unlock phone, higher in MS patients"
        )
        
        tapping = st.number_input(
            "Tapping Task (taps/10s)", 
            min_value=15.0, max_value=35.0, value=25.0, step=1.0,
            help="Number of taps in 10 seconds, lower in MS patients"
        )
        
        sleep = st.number_input(
            "Sleep Duration (hours)", 
            min_value=5.0, max_value=9.0, value=7.0, step=0.1,
            help="Daily sleep duration, more variable in MS patients"
        )

    if st.button("Predict Diagnosis"):
        try:
            # Prepare input data
            input_data = {
                'Age': age,
                'Gender': gender,
                'Heart_Rate': heart_rate,
                'Respiration_Rate': respiration_rate,
                'Skin_Temperature': skin_temp,
                'Blood_Pulse_Wave': blood_pulse,
                'Heart_Rate_Variability': hrv,
                'Galvanic_Skin_Response': gsr,
                'Energy_Expenditure': energy,
                'Steps': steps,
                'Phone_Lock_Unlock': phone_locks,
                'Tapping_Task': tapping,
                'Sleep_Duration': sleep
            }
            
            # Create properly formatted input DataFrame
            input_df = create_input_dataframe(input_data, feature_names)
            
            # Scale the input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Display results
            st.header("Diagnosis Results")
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.subheader("Prediction")
                if prediction[0] == 1:
                    st.error("⚠️ Positive MS Diagnosis")
                else:
                    st.success("✅ Negative MS Diagnosis")
            
            with col5:
                st.subheader("Confidence Score")
                confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
                st.info(f"{confidence:.2%}")
            
            # Feature importance visualization
            st.subheader("Feature Importance Analysis")
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(feature_importance, 
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Feature Importance in Diagnosis')
            st.plotly_chart(fig)
            
            # Add detailed medical disclaimer
            st.warning("""
            **Medical Disclaimer**: 
            1. This prediction is based on a machine learning model and should not be used as the sole basis for medical decisions.
            2. Multiple Sclerosis diagnosis requires comprehensive clinical evaluation, MRI scans, and other specialized tests.
            3. Please consult with a qualified healthcare provider for proper diagnosis and treatment.
            4. This tool is for research and educational purposes only.
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()