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

st.title("🇧🇷 Sistema de Previsão de Preço de Carros (FIPE)")
st.markdown("Avaliando o preço médio de carros no Brasil, baseado nos dados da Tabela FIPE.")
st.markdown("---")

FILE_NAME = "fipe_2022_december.csv"
REFERENCE_YEAR = 2025

@st.cache_data
def carregar_e_pre_processar_dados(file_name, reference_year):
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        return None

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

    categorical_cols = ['brand', 'model', 'fuel', 'gear']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df_final = df_encoded.drop(columns=['year_model'], errors='ignore')

    return df_final, df

df_final, df_raw = carregar_e_pre_processar_dados(FILE_NAME, REFERENCE_YEAR)

if df_final is None:
    st.error(f"Erro: O arquivo '{FILE_NAME}' não foi encontrado. Verifique o upload.")
    st.stop()

X = df_final.drop('Selling_Price', axis=1)
y = df_final['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.header("Configure o Veículo (FIPE)")

brands = sorted(df_raw['brand'].unique())
fuels = sorted(df_raw['fuel'].unique())
gears = sorted(df_raw['gear'].unique())
top_models = df_raw['model'].value_counts().nlargest(100).index.tolist()
models = sorted(df_raw['model'].unique()) 

def obter_features_usuario():
    
    selected_brand = st.sidebar.selectbox("Marca", brands)
    
    models_for_brand = sorted(df_raw[df_raw['brand'] == selected_brand]['model'].unique())
    selected_model = st.sidebar.selectbox("Modelo", models_for_brand)
    
    selected_fuel = st.sidebar.selectbox("Combustível", fuels)
    selected_gear = st.sidebar.selectbox("Câmbio", gears)

    min_year = int(df_raw['year_model'].min())
    max_year = int(df_raw['year_model'].max())
    selected_year = st.sidebar.number_input("Ano Modelo", min_year, max_year, max_year - 5)
    
    min_engine = df_raw['engine_size'].min()
    max_engine = df_raw['engine_size'].max()
    avg_engine = df_raw['engine_size'].mean()
    
    selected_engine_size = st.sidebar.number_input(f"Tamanho do Motor (em L, min {min_engine:.1f}, max {max_engine:.1f})", float(min_engine), float(max_engine), float(avg_engine), step=0.1)

    age_years = REFERENCE_YEAR - selected_year
    
    input_data = pd.DataFrame(np.zeros((1, X_train.shape[1])), columns=X_train.columns)
    
    input_data['age_years'] = age_years
    input_data['engine_size'] = selected_engine_size
    
    brand_col = f'brand_{selected_brand}'
    if brand_col in input_data.columns:
        input_data[brand_col] = 1

    model_col = f'model_{selected_model}'
    if model_col in input_data.columns:
        input_data[model_col] = 1
        
    fuel_col = f'fuel_{selected_fuel}'
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1
        
    gear_col = f'gear_{selected_gear}'
    if gear_col in input_data.columns:
        input_data[gear_col] = 1

    return input_data

input_df = obter_features_usuario()

st.subheader("1. Visão Geral dos Dados (FIPE BRL)")
with st.expander("Ver dataset e features (colunas) usados no treino"):
    st.markdown("Este dataset contém a média de preços da FIPE para o mês de Dezembro de 2022, já em Reais.")
    st.write(df_raw[['brand', 'model', 'fuel', 'gear', 'engine_size', 'year_model', 'Selling_Price']].head())
    st.write(f"Total de Registros (FIPE): **{df_final.shape[0]}**")
    st.write(f"Total de Features (após OHE): **{df_final.shape[1] - 1}**")


st.subheader("2. Performance e Previsão do Modelo")

model_choice = st.selectbox("Escolha o Algoritmo:", ["Random Forest", "Linear Regression"])

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

try:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    user_prediction = model.predict(input_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score (Teste)", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("MAE (Erro Médio)", f"R$ {mean_absolute_error(y_test, y_pred):,.2f}")
    col3.metric("RMSE (Erro Quadrático)", f"R$ {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

    st.success(f"### 💰 Preço Médio FIPE Previsto: R$ {user_prediction[0]:,.2f}")
    st.warning("Lembre-se: Este é o preço médio FIPE. O valor de venda real varia muito com a quilometragem, estado e opcionais.")


    st.subheader("3. Visualização de Resultados")

    st.markdown("#### Preço Real (Teste) vs. Preço Previsto")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    lims = (min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
    ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax1.set_xlabel("Preço Médio FIPE Real (R$)")
    ax1.set_ylabel("Preço Médio FIPE Previsto (R$)")
    ax1.set_title("Real vs. Previsto")
    st.pyplot(fig1)

    if model_choice == "Random Forest":
        st.markdown("#### Fatores que Mais Influenciam o Preço (Importância de Feature)")
        
        feature_importance_df = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
        
        feature_importance_df = feature_importance_df[feature_importance_df > 0] 
        
        def limpar_nome_feature(name):
            if name.startswith('brand_'): return f"Marca: {name[6:]}"
            if name.startswith('model_'): return f"Modelo: {name[6:]}"
            if name.startswith('fuel_'): return f"Comb.: {name[5:]}"
            if name.startswith('gear_'): return f"Câmbio: {name[5:]}"
            if name == 'age_years': return "Idade do Carro"
            if name == 'engine_size': return "Tamanho do Motor (L)"
            return name

        feature_importance_df.index = feature_importance_df.index.map(limpar_nome_feature)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        feature_importance_df.plot(kind='barh', ax=ax2)
        ax2.set_title("Top 20 Fatores Mais Importantes")
        ax2.set_xlabel("Importância")
        st.pyplot(fig2)

except Exception as e:
    st.error(f"Erro durante o treinamento ou previsão do modelo: {e}")
    st.warning("Verifique se as colunas categóricas selecionadas na sidebar têm correspondência nos dados de treinamento após o One-Hot Encoding.")


st.markdown("---")
st.caption("Desenvolvido para análise de regressão de preços FIPE.")
