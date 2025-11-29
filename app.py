import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuração da Página
st.set_page_config(page_title="Prevendo Preço de Carros", layout="wide")

st.title("🚗 Sistema de Previsão de Preço de Carros (BRL)")
st.markdown("---")

# 1. Carregamento e Conversão de Moeda
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("car data.csv")
        # --- CONVERSÃO DE MOEDA (O Pulo do Gato) ---
        # 1 Lakh Indiano ~= 7.000 Reais
        taxa_conversao = 7000
        
        # Convertendo as colunas de preço para Reais
        df['Selling_Price'] = df['Selling_Price'] * taxa_conversao
        df['Present_Price'] = df['Present_Price'] * taxa_conversao
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Erro: O arquivo 'car data.csv' não foi encontrado. Verifique o upload.")
    st.stop()

# 2. Pré-processamento
replace_dict = {
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Seller_Type': {'Dealer': 0, 'Individual': 1},
    'Transmission': {'Manual': 0, 'Automatic': 1}
}

df_encoded = df.replace(replace_dict)
df_encoded['Age'] = 2024 - df_encoded['Year']
df_final = df_encoded.drop(['Car_Name', 'Year'], axis=1)

X = df_final.drop('Selling_Price', axis=1)
y = df_final['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SIDEBAR (Entrada do Usuário) ---
st.sidebar.header("Configure o Carro")

def user_input_features():
    # Agora os inputs são em Reais, valores maiores
    Present_Price = st.sidebar.number_input("Preço de Tabela (0km) em R$", 10000.0, 1000000.0, 35000.0)
    Kms_Driven = st.sidebar.number_input("Kilômetros Rodados", 0, 500000, 20000)
    
    Fuel_Type_Label = st.sidebar.selectbox("Combustível", ['Petrol', 'Diesel', 'CNG'])
    Seller_Type_Label = st.sidebar.selectbox("Tipo de Vendedor", ['Dealer', 'Individual'])
    Transmission_Label = st.sidebar.selectbox("Câmbio", ['Manual', 'Automatic'])
    Owner = st.sidebar.selectbox("Donos Anteriores", [0, 1, 3])
    Year = st.sidebar.number_input("Ano de Fabricação", 2000, 2024, 2015)
    
    data = {
        'Present_Price': Present_Price,
        'Kms_Driven': Kms_Driven,
        'Fuel_Type': replace_dict['Fuel_Type'][Fuel_Type_Label],
        'Seller_Type': replace_dict['Seller_Type'][Seller_Type_Label],
        'Transmission': replace_dict['Transmission'][Transmission_Label],
        'Owner': Owner,
        'Age': 2024 - Year
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- ÁREA PRINCIPAL ---

st.subheader("1. Visão Geral dos Dados (Convertidos para BRL)")
with st.expander("Ver dataset"):
    st.write(df.head())

st.subheader("2. Performance do Modelo")

model_choice = st.selectbox("Escolha o Algoritmo:", ["Linear Regression", "Random Forest"])

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
user_prediction = model.predict(input_df)

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
# Exibindo erro em Reais
col2.metric("MAE (Erro Médio)", f"R$ {mean_absolute_error(y_test, y_pred):,.2f}")
col3.metric("RMSE (Erro Quadrático)", f"R$ {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

st.success(f"### 💰 Preço Previsto: R$ {user_prediction[0]:,.2f}")

# --- GRÁFICOS ---
st.subheader("3. Visualização")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Preço Real (R$)")
plt.ylabel("Preço Previsto (R$)")
plt.title("Real vs. Previsto")
st.pyplot(fig)

if model_choice == "Random Forest":
    st.info("Fatores que mais influenciam o preço:")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importance)