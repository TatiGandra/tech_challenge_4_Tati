import streamlit as st
import pandas as pd
from prophet import Prophet  # Importe o Prophet
import json
from prophet.serialize import model_to_json, model_from_json
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Carregue os dados e o modelo
dados = pd.read_csv('https://raw.githubusercontent.com/TatiGandra/tech_challenge_4_Tati/main/df_clean.csv')

# Criar guias
titulos_guias = ['Introdução','Simulador', 'DashBoard']
guia1, guia2, guia3 = st.tabs(titulos_guias)

with guia1:
    st.header('Petróleo Brent')
    
    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
     
    st.subheader('Introdução')

    st.markdown("""
    
    Bem-vindo a análise dos dados do petróleo brent, que é uma classificação de petróleo cru originário do mar do norte.

    Para este trabalho, importamos a base histórica de preços encontrado na base de dados do Ipea e criamos um simulador de preços deste produto e um dashboard com análises desses mesmos dados e variações geopolíticas que os afetaram.

    Convidamos você a se juntar a nós nesta análise. Acreditamos que você encontrará esta análise informativa e útil. 
    
    Vamos começar!
    """)

    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
    
    st.subheader('Pós Tech - FIAP')
    
    st.markdown('Created by:')       
    st.markdown('Leandro Castro - RM 350680')
    st.markdown('Mateus Correa - RM 351094')
    st.markdown('Tatiane Gandra - RM 352177' )
 
    with guia2:
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
            st.write('O preço do petróleo brent previsto para esse dia será de U$', round(previsao['yhat'].iloc[0],2))
            dados['ds'] = pd.to_datetime(dados['ds'])

            # Filtrar os dados para o primeiro trimestre de 2024
            primeiro_trimestre_2024 = dados[(dados['ds'] >= '2024-01-01') & (dados['ds'] <= '2024-03-31')]

            # Calcular a média dos valores do primeiro trimestre de 2024
            media_primeiro_trimestre = primeiro_trimestre_2024['y'].mean()

            # Calcular a previsão (este é um exemplo, substitua pelo seu código de previsão)
            # Exemplo:
            # nova_data = pd.DataFrame({'ds': pd.date_range(start='2024-05-01', periods=30, freq='D')})
            # previsao = modelo.predict(nova_data)

            # Supondo que 'previsao' é um DataFrame que contém as previsões
            # Supondo que você quer comparar a média do primeiro trimestre com a primeira previsão
            valor_previsto = previsao['yhat'].iloc[0]  # ou o índice que você quiser
            percentual_diferenca = ((valor_previsto - media_primeiro_trimestre) / media_primeiro_trimestre) * 100
            # Comparar a previsão com a média do primeiro trimestre de 2024
            if valor_previsto > media_primeiro_trimestre:
                comparacao = "maior"
            else:
                comparacao = "menor"

            # Formatar a previsão e o percentual com duas casas decimais
            valor_previsto_formatado = round(valor_previsto, 2)
            percentual_diferenca_formatado = round(percentual_diferenca, 2)

            # Exibir a previsão e a comparação
            st.write(f"Ele é {comparacao} do que a média do primeiro trimestre de 2024 ({media_primeiro_trimestre:.2f}). Isto representa uma diferença de {percentual_diferenca_formatado}%.")

        
        with guia3:
            st.header('Time Series - Petróleo')

            # Criar o gráfico de linha
            trace = go.Scatter(x=dados['ds'], y=dados['y'], mode='lines', name='Preço do Petróleo')

            # Criar o layout do gráfico
            layout = go.Layout(title='Time Series - Petróleo', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'), hovermode='closest')

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            st.plotly_chart(fig)
            st.write('''
                     * Flutuações Significativas: O gráfico mostra uma flutuação significativa nos preços do petróleo, com quedas acentuadas e aumentos notáveis em certos pontos.
                     * Estabilidade e Queda em 2020: Inicialmente, em 2020, o preço estava estável em torno de $60 por barril antes de cair drasticamente.
                     * Volatilidade: Os picos visíveis no gráfico indicam períodos de rápida mudança de preços, refletindo eventos que impactaram o mercado de petróleo.
                     * Mercado Volátil: As flutuações ao longo dos anos mostram um mercado volátil, com variações de preço que podem ser influenciadas por fatores geopolíticos, demanda global e outros fatores econômicos.
                     
                     Este gráfico de linha do tempo é uma ferramenta valiosa para analisar o comportamento do mercado de petróleo ao longo de um período de cinco anos, permitindo identificar tendências e potenciais padrões no movimento dos preços.
                     ''')
            
            st.header('Média Mensal')
            dados['ds'] = pd.to_datetime(dados['ds'])
            dados.set_index('ds', inplace=True)
            # Calcular a média mensal dos preços do petróleo
            monthly_avg = dados.resample('M').mean()

            # Criar o gráfico de barras
            trace = go.Bar(x=monthly_avg.index, y=monthly_avg['y'], marker=dict(color='skyblue'), name='Média Mensal')

            # Criar o layout do gráfico
            layout = go.Layout(title='Média Mensal de Preços do Petróleo', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)

            st.write('''
                    * Variação Significativa: O gráfico mostra que houve uma variação significativa nos preços ao longo dos cinco anos, com mudanças mensais evidentes.
                    * Picos em 2022: Há picos notáveis em 2022, sugerindo um aumento considerável nos preços médios mensais nesse ano.
                    * Valor da Média Mensal: O eixo Y, rotulado como “Valor da Média Mensal em Reais”, mostra que os preços variaram de 0 a 120 reais.
                    * Análise Temporal: O eixo X, que representa a data de 2020 a 2024, permite visualizar a tendência dos preços ao longo do tempo, destacando meses específicos com preços médios mais altos.

                    Essas informações podem ser cruciais para entender a tendência do mercado de petróleo e para tomar decisões informadas relacionadas a investimentos e estratégias econômicas.
                    '''
            )
            st.header('Histograma')
            # Criar o histograma dos preços do petróleo
            trace = go.Histogram(x=dados['y'], marker=dict(color='skyblue'))

            # Criar o layout do gráfico
            layout = go.Layout(title='Distribuição dos Preços do Petróleo', xaxis=dict(title='Valor de Fechamento'), yaxis=dict(title='Contagem'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)
            st.write('''
                    * Faixa de Preço Mais Comum: A maior frequência de preços está entre 60 a 80 dólares, indicando que esta foi a faixa de preço mais observada.
                    * Variação de Preço: Existem ocorrências significativas na faixa de 80 a 100 dólares, mostrando que os preços também alcançaram valores mais altos com certa regularidade.
                    * Preços Menos Comuns: As faixas de preço abaixo de 20 e acima de 100 dólares têm menos ocorrências, sugerindo que são eventos menos comuns.
                    * Distribuição: O histograma mostra uma distribuição variada dos preços, com a maior parte concentrada na faixa média de preço.
                    
                    O histograma fornece uma visão geral útil da volatilidade dos preços do petróleo e das faixas de preço mais frequentes ao longo dos últimos cinco anos. Isso pode ser útil para análises econômicas e previsões de mercado.
                    '''
            )
            st.header('Boxplot')
            # Criar o boxplot dos preços do petróleo
            trace = go.Box(y=dados['y'], marker=dict(color='skyblue'))

            # Criar o layout do gráfico
            layout = go.Layout(title='Dispersão dos Preços do Petróleo', yaxis=dict(title='Valor de Fechamento'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)
            st.write('''
                    * Mediana: A linha dentro da caixa principal indica a mediana dos preços, que é o valor central da distribuição.
                    * Intervalo Interquartil: A caixa em si representa o intervalo interquartil, mostrando onde a maioria dos dados está concentrada.
                    * Variação dos Preços: Os “bigodes” do boxplot se estendem para mostrar a variação total dos preços dentro de 1,5 vezes o intervalo interquartil.
                    * Uniformidade dos Dados: A ausência de outliers sugere que os preços do petróleo foram relativamente estáveis, sem flutuações extremas.

                    Este boxplot fornece uma visão clara da distribuição dos preços do petróleo, destacando a mediana e a consistência dos preços ao longo do tempo.
            ''')
            st.header('Densidade')
            # Criar o gráfico de densidade dos preços do petróleo
            trace = go.Scatter(x=dados.index, y=dados['y'], mode='lines', fill='tozeroy', fillcolor='skyblue', line=dict(color='blue'))

            # Criar o layout do gráfico
            layout = go.Layout(title='Densidade dos Preços do Petróleo', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)
            st.write('''
            * Flutuações Visíveis: O gráfico mostra flutuações notáveis nos preços do petróleo, com picos e quedas significativos ao longo do período.
            * Picos Significativos: Entre 2021 e 2022, observam-se picos acentuados, indicando um aumento abrupto nos preços.
            * Variação de Preços: O eixo Y, que representa a ‘Valor da Densidade dos Preços do Petróleo’, varia de 0 a 140, mostrando uma ampla gama de variação nos preços.
            * Análise Temporal: O eixo X cobre de 2020 a 2024, permitindo uma análise da evolução dos preços ao longo do tempo.

            Este gráfico é útil para entender como a densidade dos preços do petróleo mudou e pode ajudar a identificar tendências ou padrões no mercado de petróleo.
            ''')