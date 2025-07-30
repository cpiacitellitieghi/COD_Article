# COD_Article# Artigo: Previsão de Irradiação Solar utilizando Inteligência Artificial em Dados Meteorológicos

Autores:  

**Guilherme Nunes Naufal**  
Escola de Artes, Ciências e Humanidades (EACH), Universidade de São Paulo (USP), São Paulo, Brasil  
guilherme.naufal@usp.br  

**Camila Piacitelli Tieghi**  
Faculdade de Tecnologia de Botucatu (FATEC), Centro Paula Souza, Botucatu, Brasil  
camila.tieghi@fatec.sp.gov.br  

**Fernando de Lima Caneppele**  
Faculdade de Zootecnia e Engenharia de Alimentos (FZEA), Universidade de São Paulo (USP), Pirassununga, Brasil  
fernando.caneppele@usp.br  

---

## Descrição do Projeto  

Este repositório contém a implementação de modelos de aprendizado de máquina para previsão de irradiação solar diária na cidade de São Paulo, utilizando dados meteorológicos do sistema NASA POWER. O projeto inclui scripts em Python para pré-processamento, modelagem, avaliação e visualização dos resultados, com foco em aplicações para energia renovável e gestão urbana sustentável.  

---

## Arquivos Incluídos  

### `irradiação_solar_EACH-USP.py`  
**Descrição:** Script principal em Python que implementa o fluxo completo de análise, desde a carga dos dados até a avaliação de modelos de regressão para previsão de irradiação solar.  
**Funcionalidades:**  
- Pré-processamento de dados meteorológicos (normalização, tratamento de valores faltantes).  
- Implementação de seis modelos de regressão: Regressão Linear, Random Forest, SVM, Rede Neural (MLP), Gradient Boosting e Árvore de Decisão (CART).  
- Cálculo de métricas de desempenho (MSE, RMSE, MAE, R²).  
- Geração de gráficos comparativos, matriz de correlação, curvas de aprendizado e análise de resíduos.  

### `data/POWER_Point_Daily_20150101_20250101_023d48S_046d50W_LST.json`  
**Descrição:** Arquivo JSON contendo dados meteorológicos diários de São Paulo (2015-2025), coletados pelo sistema NASA POWER.  
**Conteúdo:**  
- Variáveis climáticas: temperatura, umidade, vento, pressão, precipitação e radiação solar.  
- Variável alvo: irradiação solar na superfície (`ALLSKY_SFC_SW_DWN`).  

### `resultados/`  
**Descrição:** Diretório com saídas geradas pelo script, incluindo:  
- Tabelas de métricas comparativas (RMSE, MAE, R²).  
- Gráficos de dispersão (valores reais vs. previstos).  
- Séries temporais comparando previsões e dados reais.  
- Matriz de correlação entre grupos de variáveis meteorológicas.  

---

## Métodos Utilizados  

1. **Pré-processamento:**  
   - Substituição de valores faltantes (`-999`) pela média.  
   - Normalização dos dados usando `MinMaxScaler`.  
   - Divisão dos dados em conjuntos de treino (80%) e teste (20%).  

2. **Modelagem:**  
   - **Random Forest e Gradient Boosting:** Modelos ensemble com ajuste de hiperparâmetros para capturar relações não lineares.  
   - **Rede Neural (MLP):** Arquitetura com camadas ocultas (100, 50) e taxa de aprendizado de 0.001.  
   - **Validação Cruzada:** Avaliação de robustez com *k-fold* (k=3).  

3. **Análise:**  
   - **Seleção de Características:** Uso de `SelectKBest` para identificar variáveis mais relevantes.  
   - **Correlação:** Agrupamento de variáveis em categorias (temperatura, umidade, etc.) para análise simplificada.  

---

## Resultados Destacados  

- **Melhores Modelos:** Random Forest e Gradient Boosting alcançaram R² > 0.91 e RMSE < 0.52 kWh/m²/dia.  
- **Consistência Física:** Correlações identificadas (ex.: radiação vs. temperatura: +0.57) alinhadas com a literatura.  
- **Visualizações:**  
  - Série temporal da irradiação solar (2015-2025) com sazonalidade evidente.  
  - Histogramas de resíduos para análise de viés.  

---

## Instruções para Uso  

1. **Pré-requisitos:**  
   - Python 3.x, bibliotecas: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.  

2. **Execução:**  
   ```bash  
   python irradiação_solar_EACH-USP.py  
   ```  

3. **Saídas:**  
   - Métricas salvas em `resultados/metricas.csv`.  
   - Gráficos salvos em `resultados/graficos/`.  

---

## Contribuições  

Este trabalho foi desenvolvido no âmbito do programa *Energia Sustentável da USP*, com apoio da FAPESP. Os resultados contribuem para a integração de energias renováveis em ambientes urbanos, alinhando-se com os Objetivos de Desenvolvimento Sustentável (ODS 7).  

--- 

**Licença:** Creative Commons Attribution 4.0 International.  
**Contato:** fernando.caneppele@usp.br
