import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

dados = pd.read_csv('C:\Users\Inteli\Desktop\Pos Tech modulo 4\tech-challenge-4\df_clean.csv')

st.write('# Simulador de avaliação de crédito de preço do petróleo brent')