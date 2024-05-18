import streamlit as st
import pandas as pd
from prophet import Prophet  # Importe o Prophet
import json
from prophet.serialize import model_to_json, model_from_json

# Carregue os dados e o modelo
dados = pd.read_csv('https://raw.githubusercontent.com/TatiGandra/tech_challenge_4_Tati/main/df_clean.csv')

with open('modelo/prophet_model.json', 'r') as f:
    model_json = json.load(f)
    modelo_carregado = model_from_json(model_json)


# Permita que o usuário selecione a data desejada
st.write('# Simulador de preço do petróleo brent')
input_data = st.date_input('Selecione uma data', format="YYYY-MM-DD")

# Crie um DataFrame com a data selecionada pelo usuário
nova_data = pd.DataFrame({'ds': [input_data]})

# Faça a previsão para a data selecionada
previsao = modelo_carregado.predict(nova_data)

# Exiba o resultado da previsão para o usuário
if st.button('Enviar'):
    st.write('O preço do petróleo brent nesse dia será de U$', round(previsao['yhat'].iloc[0],2))
