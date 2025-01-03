import streamlit as st
import pandas as pd

st.set_page_config(page_title="Classifier Performance Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    basic = pd.read_csv('classifier_basic_results.csv')
    detailed = pd.read_csv('classifier_detailed_results.csv')
    return basic, detailed

basic_results, detailed_models = load_data()

st.title("Machine Learning Classifier Performance Dashboard")

# Basic Results Section
st.header("Basic Classifier Results")
col1, col2 = st.columns(2)

with col2:
    # Table view
    st.dataframe(basic_results.style.highlight_max(subset=['Accuracy']))

# Detailed Results Section
st.header("Detailed Model Performance")

# Model Selection and Detailed View
selected_model = st.selectbox('Select Model for Detailed View', detailed_models['Model'])
model_data = detailed_models[detailed_models['Model'] == selected_model].iloc[0]

col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Accuracy", f"{model_data['Accuracy']:.3f}")
with col4:
    st.metric("Positive Class F1", 
              f"{2 * (model_data['Positive_Precision'] * model_data['Positive_Recall']) / (model_data['Positive_Precision'] + model_data['Positive_Recall']):.3f}")
with col5:
    st.metric("Negative Class F1", 
              f"{2 * (model_data['Negative_Precision'] * model_data['Negative_Recall']) / (model_data['Negative_Precision'] + model_data['Negative_Recall']):.3f}")
