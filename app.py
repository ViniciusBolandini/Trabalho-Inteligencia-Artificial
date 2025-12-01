import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prevendo Preço de Carros (FIPE)", layout="wide")

st.title("Sistema de Previsão de Preço de Carros (FIPE)")
st.markdown("Avaliando o preço médio de carros no Brasil, baseado nos dados da Tabela FIPE.")
st.markdown("---")

FILE_NAME = "fipe_2022_december.csv"
REFERENCE_YEAR = 2025

@st.cache_data
def carregar_e_pre_processar_dados(file_name, reference_year):
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        return None, None, None

    df = df.drop(columns=['year_of_reference', 'month_of_reference', 'fipe_code', 'authentication'], errors='ignore')
    df = df.rename(columns={'avg_price_brl': 'Selling_Price'})
    
    df['engine_size'] = pd.to_numeric(df['engine_size'], errors='coerce')
    df['Selling_Price'] = pd.to_numeric(df['Selling_Price'], errors='coerce')
    df['age_years'] = reference_year - df['year_model']
    
    for col in ['brand', 'model', 'fuel', 'gear']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    if 'engine_size' in df.columns:
        df['engine_size'] = df['engine_size'].fillna(df['engine_size'].mean())

    top_models = df['model'].value_counts().nlargest(50).index.tolist()
    
    df_for_encoding = df.copy()
    df_for_encoding['model'] = df_for_encoding['model'].apply(lambda x: x if x in top_models else 'Other')

    categorical_cols = ['brand', 'model', 'fuel', 'gear']
    df_encoded = pd.get_dummies(df_for_encoding, columns=categorical_cols, drop_first=True)

    df_final = df_encoded.drop(columns=['year_model'], errors='ignore')

    return df_final, df, top_models

df_final, df_raw, top_models_list = carregar_e_pre_processar_dados(FILE_NAME, REFERENCE_YEAR)

if df_final is None:
    st.error(f"Erro: O arquivo '{FILE_NAME}' não foi encontrado. Verifique o upload.")
    st.stop()

X = df_final.drop('Selling_Price', axis=1)
y = df_final['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def treinar_modelo(tipo_modelo, _X_train, _y_train):
    if tipo_modelo == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=40, max_depth=15, random_state=42, n_jobs=-1)
    
    model.fit(_X_train, _y_train)
    return model

st.sidebar.header("Configure o Veículo (FIPE)")

brands = sorted(df_raw['brand'].unique())
fuels = sorted(df_raw['fuel'].unique())
gears = sorted(df_raw['gear'].unique())

selected_brand = st.sidebar.selectbox("Marca", brands, index=None, placeholder="Selecione a marca...")

if selected_brand:
    models_for_brand = sorted(df_raw[df_raw['brand'] == selected_brand]['model'].unique())
    selected_model = st.sidebar.selectbox("Modelo", models_for_brand, index=None, placeholder="Selecione o modelo...")
else:
    selected_model = None
    st.sidebar.selectbox("Modelo", [], disabled=True, placeholder="Primeiro selecione a marca")

selected_fuel = st.sidebar.selectbox("Combustível", fuels)
selected_gear = st.sidebar.selectbox("Câmbio", gears)

min_year = int(df_raw['year_model'].min())
max_year = int(df_raw['year_model'].max())
selected_year = st.sidebar.number_input("Ano Modelo", min_year, max_year, max_year - 5)

min_engine = df_raw['engine_size'].min()
max_engine = df_raw['engine_size'].max()
avg_engine = df_raw['engine_size'].mean()
selected_engine_size = st.sidebar.number_input(f"Tamanho do Motor (L)", float(min_engine), float(max_engine), float(avg_engine), step=0.1)

if not selected_brand or not selected_model:
    st.info("Comece pela Barra Lateral: Selecione a Marca e o Modelo para ver a previsão.")
    st.stop()

def preparar_input_usuario():
    age_years = REFERENCE_YEAR - selected_year
    input_data = pd.DataFrame(np.zeros((1, X_train.shape[1])), columns=X_train.columns)
    
    input_data['age_years'] = age_years
    input_data['engine_size'] = selected_engine_size
    
    brand_col = f'brand_{selected_brand}'
    if brand_col in input_data.columns: input_data[brand_col] = 1

    if selected_model in top_models_list:
        model_col = f'model_{selected_model}'
        if model_col in input_data.columns: input_data[model_col] = 1
    else:
        if 'model_Other' in input_data.columns: input_data['model_Other'] = 1
        
    fuel_col = f'fuel_{selected_fuel}'
    if fuel_col in input_data.columns: input_data[fuel_col] = 1
        
    gear_col = f'gear_{selected_gear}'
    if gear_col in input_data.columns: input_data[gear_col] = 1

    return input_data

input_df = preparar_input_usuario()

st.subheader("1. Visão Geral dos Dados (FIPE)")
with st.expander("Ver detalhes do Dataset"):
    st.markdown("O modelo foi otimizado para considerar os 50 carros mais populares individualmente e agrupar o restante em categorias gerais para economizar memória.")
    st.write(f"Total de Registros: **{df_final.shape[0]}**")
    st.write(f"Total de Features (Colunas): **{df_final.shape[1] - 1}**")

st.subheader("2. Performance e Previsão")

model_choice = st.selectbox("Escolha o Algoritmo:", ["Random Forest", "Linear Regression"])

try:
    with st.spinner('Processando...'):
        model = treinar_modelo(model_choice, X_train, y_train)
    
    y_pred = model.predict(X_test)
    user_prediction = model.predict(input_df)
    valor_previsto = user_prediction[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("MAE", f"R$ {mean_absolute_error(y_test, y_pred):,.2f}")
    col3.metric("RMSE", f"R$ {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

    if valor_previsto < 0:
        st.error(f"Estimativa Inválida: R$ {valor_previsto:,.2f}")
        st.markdown("**Motivo:** A Regressão Linear gerou um valor negativo para este cenário (limitação matemática do modelo). Use o Random Forest.")
    else:
        st.success(f"Preço FIPE Previsto: R$ {valor_previsto:,.2f}")

    st.subheader("3. Visualização")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_test[:1000], y=y_pred[:1000], ax=ax1, alpha=0.5)
    lims = (min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
    ax1.plot(lims, lims, 'r--')
    ax1.set_xlabel("Real (R$)")
    ax1.set_ylabel("Previsto (R$)")
    st.pyplot(fig1)

except Exception as e:
    st.error(f"Erro: {e}")

st.markdown("---")
st.caption("Sistema Otimizado para Streamlit Cloud (Low Memory Mode)")