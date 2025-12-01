# Previsão de preço de veículos brasileiros: Regressão

## 🔗 Acesso à Aplicação (Streamlit)

A aplicação de predição pode ser acessada e testada em tempo real através do link abaixo:

**[Acessar a Aplicação de Predição](https://trabalho-inteligencia-artificial-khchq2tofckk9wpxpr8jzt.streamlit.app/)**

---

## 💡 Problema e Objetivo

### Problema

O mercado de veículos usados no Brasil enfrenta volatilidade e subjetividade na precificação. A dependência de referências genéricas resulta em estimativas inconsistentes, gerando ineficiência nas transações. O projeto busca mitigar essa incerteza, desenvolvendo um modelo robusto para prever o valor de mercado de um veículo com alta acurácia.

### Objetivo

O objetivo é criar uma solução inteligente capaz de converter as características do veículo (marca, modelo, ano, tipo de combustível, câmbio e tamanho do motor em L) em uma **estimativa precisa do preço de venda**, auxiliando na definição de um preço justo de mercado.

---

## 📚 Metodologia e Modelagem (Machine Learning)

O projeto utilizou a biblioteca **scikit-learn (sklearn)** em Python para implementar e comparar dois algoritmos de regressão:

### Algoritmos Testados:

| Modelo | Vantagens | Desafios Encontrados |
| :--- | :--- | :--- |
| **Regressão Linear (`LinearRegression`)** | Simples e de alta interpretabilidade. | Falha em modelar a não-linearidade do mercado, resultando em **extrapolação falha** (previsão de valores negativos para carros muito antigos). |
| **Random Forest Regressor (`RandomForestRegressor`)** | Alta robustez e capacidade de modelar relações não-lineares complexas. | **Modelo Vencedor.** Apresentou previsões muito mais consistentes, sendo mais adequado para o domínio de preços de veículos. |

---

## 💾 Conjunto de Dados (Dataset)

A escolha do dataset foi um ponto crucial de evolução no projeto:

1.  **Dataset Inicial (Descartado):** *Vehicle Dataset from Cardekho* (Kaggle). Embora contivesse dados valiosos para veículos usados (quilometragem e donos), era **indiano** e possuía **volume limitado**, distorcendo os preços e modelos para a realidade brasileira.
2.  **Dataset Final (Adotado):** ***Average Car Prices - Brazil*** (Kaggle). Apesar de perder algumas variáveis de histórico de uso, a mudança garantiu a **relevância geográfica** e um **volume robusto** de mais de 20 mil registros, tornando os dados muito mais condizentes com o mercado brasileiro.

---

## 📈 Métricas de Desempenho

A performance do modelo foi avaliada utilizando três métricas essenciais de Regressão:

| Métrica | Descrição | Importância |
| :--- | :--- | :--- |
| **$R^2$ Score** | Coeficiente de Determinação. | Indica a capacidade explicativa do modelo (percentual da variação de preços explicada pelas *features*). |
| **MAE (Erro Médio Absoluto)** | Média da diferença absoluta (em Reais) entre o valor previsto e o valor real. | **Métrica de Negócios.** Representa o erro monetário médio esperado do modelo em uma previsão. |
| **RMSE (Raiz do Erro Quadrático Médio)** | Raiz quadrada da média dos erros quadrados. | **Métrica de Robustez.** Penaliza erros grandes de forma desproporcional, crucial para evitar previsões financeiramente "absurdas" (outliers). |

---

## 🚀 Trabalhos Futuros

Para aprimorar a precisão e a utilidade do sistema, as próximas etapas incluem:

1.  **Expansão do Dataset (Prioridade):** Integrar **dados de histórico de uso** (quilometragem e número de proprietários) para aumentar significativamente a acurácia.
2.  **Inclusão de Variáveis Contextuais:** Adicionar fatores externos como **indicadores macroeconômicos** (inflação, taxa SELIC) e **dados de liquidez regional** para refletir as dinâmicas temporais do mercado automotivo.
