import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import os

warnings.filterwarnings("ignore")

# Caminho do arquivo
file_path = 'data\POWER_Point_Daily_20150101_20250101_023d48S_046d50W_LST (1).json'

# Carregar os dados
try:
    with open(file_path) as f:
        dados = json.load(f)
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado. Verifique o caminho e tente novamente.")
    exit()

# Converter os dados para DataFrame
parametros = dados['properties']['parameter']
df = pd.DataFrame(parametros)

# Índice como datas
df.index = pd.to_datetime(df.index, format='%Y%m%d')
df.sort_index(inplace=True)

print("\nDados carregados com sucesso:")
print(df.head())

# Tratar valores faltantes
df.replace(-999, np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# Definir variável alvo
target = 'ALLSKY_SFC_SW_DWN'

# Separar preditores e alvo
X = df.drop(target, axis=1)
y = df[target]

# Normalização
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)
y_scaled = pd.Series(scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Definir modelos com hiperparâmetros ajustados para tentar replicar as tabelas fornecidas
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42, learning_rate_init=0.001),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    'CART': DecisionTreeRegressor(max_depth=5, random_state=42)
}

# Dicionário para armazenar métricas e previsões
metrics = {'Model': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'R²': []}
predictions = {}

# Treinar e avaliar modelos
plt.figure(figsize=(15, 8))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Desnormalizar para métricas e tabelas
    y_test_real = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calcular métricas
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)

    metrics['Model'].append(name)
    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['MAE'].append(mae)
    metrics['R²'].append(r2)

    print(f"\n{name}:")
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')

    # Gráfico de dispersão
    plt.subplot(2, 3, i)
    plt.scatter(y_test_real, y_pred_real, alpha=0.5)
    plt.plot([min(y_test_real), max(y_test_real)], [min(y_test_real), max(y_test_real)], 'r--')
    plt.xlabel('Real (kWh/m²/dia)')
    plt.ylabel('Previsto (kWh/m²/dia)')
    plt.title(name)
    plt.tight_layout()
plt.show()

# Exibir métricas
metrics_df = pd.DataFrame(metrics)
print("\nMétricas dos Modelos:\n")
print(metrics_df)

# Gerar tabelas de estatísticas descritivas
def gerar_tabela_descritiva(modelo_nome, y_pred, y_test, scaler_y):
    y_real = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    stats_real = pd.Series(y_real).describe()
    stats_pred = pd.Series(y_pred_real).describe()

    tabela = pd.DataFrame({
        'Estatísticas Descritivas': ['Contagem', 'Média', 'Desvio padrão', 'Min', '25%', '50%', '75%', 'Máx'],
        'Real': [
            stats_real['count'],
            round(stats_real['mean'], 4),
            round(stats_real['std'], 4),
            round(stats_real['min'], 4),
            round(stats_real['25%'], 4),
            round(stats_real['50%'], 4),
            round(stats_real['75%'], 4),
            round(stats_real['max'], 4)
        ],
        modelo_nome: [
            stats_pred['count'],
            round(stats_pred['mean'], 4),
            round(stats_pred['std'], 4),
            round(stats_pred['min'], 4),
            round(stats_pred['25%'], 4),
            round(stats_pred['50%'], 4),
            round(stats_pred['75%'], 4),
            round(stats_pred['max'], 4)
        ]
    })

    print(f"\nTabela — Estatística Descritiva do Modelo {modelo_nome}:\n")
    print(tabela.to_string(index=False))
    return tabela

tabelas = {}
for nome_modelo, y_pred in predictions.items():
    tabela = gerar_tabela_descritiva(nome_modelo, y_pred, y_test, scaler_y)
    tabelas[nome_modelo] = tabela

# Gráficos comparativos: Real vs Predito
plt.figure(figsize=(18, 10))
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    y_test_real = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    plt.subplot(2, 3, i)
    plt.plot(y_test_real[:100], label='Valor Real', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred_real[:100], label='Valor Previsto', marker='x', linestyle='--', alpha=0.7)
    plt.title(f'{name}')
    plt.xlabel('Índice da Amostra')
    plt.ylabel('Irradiação (kWh/m²/dia)')
    plt.legend()
    plt.tight_layout()
plt.show()

# Curva de aprendizado
plt.figure(figsize=(15, 8))
for i, (name, model) in enumerate(models.items(), 1):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y_scaled, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2'
    )
    plt.subplot(2, 3, i)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Treino')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Teste')
    plt.xlabel('Tamanho Treino')
    plt.ylabel('R²')
    plt.title(f'Curva - {name}')
    plt.legend()
    plt.tight_layout()
plt.show()

# Variação temporal diária
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[target], marker='o', markersize=2)
plt.title('Variação Temporal da Irradiação Solar Diária (2015-2025)')
plt.xlabel('Data')
plt.ylabel('Irradiação (kWh/m²/dia)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Mapa de correlação agrupado
grupos = {
    'Temperatura': ['T2M', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE'],
    'Vento': ['WS2M', 'WS10M', 'WS10M_MAX'],
    'Umidade': ['RH2M', 'QV2M'],
    'Pressão/Precipitação': ['PS', 'PRECTOTCORR'],
    'Radiação/Irradiação': ['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DIFF', 'ALLSKY_KT', 'CLRSKY_SFC_SW_DWN']
}

df_grupos = pd.DataFrame()
for nome, variaveis in grupos.items():
    valid_vars = [var for var in variaveis if var in df.columns]
    if valid_vars:
        df_grupos[nome] = df[valid_vars].mean(axis=1)

corr = df_grupos.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Correlação entre Grupos de Variáveis')
plt.tight_layout()
plt.show()

# Análise de resíduos
plt.figure(figsize=(15, 8))
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    y_test_real = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    residuos = y_test_real - y_pred_real
    plt.subplot(2, 3, i)
    plt.hist(residuos, bins=20)
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.title(f'Resíduos - {name}')
    plt.tight_layout()
plt.show()

# Análise de sensibilidade para Regressão Linear
scores = []
nomes_variaveis = X.columns
for i in range(X.shape[1]):
    X_temp = X_scaled.drop(X.columns[i], axis=1)
    lr = LinearRegression()
    scores_cv = cross_val_score(lr, X_temp, y_scaled, cv=3, scoring='r2')
    scores.append(np.mean(scores_cv))

plt.figure(figsize=(12, 6))
plt.bar(nomes_variaveis, scores)
plt.xlabel('Características')
plt.ylabel('R² Médio (Validação Cruzada)')
plt.title('Análise de Sensibilidade - Regressão Linear')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Validação cruzada
plt.figure(figsize=(15, 8))
for i, (name, model) in enumerate(models.items(), 1):
    scores = cross_val_score(model, X_scaled, y_scaled, cv=3, scoring='r2')
    plt.subplot(2, 3, i)
    plt.bar(range(1, len(scores) + 1), scores)
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title(f'Validação Cruzada - {name}')
    plt.tight_layout()
plt.show()

# Seleção de características
selector = SelectKBest(f_regression, k=5)
selector.fit(X_scaled, y_scaled)
support = selector.get_support()
features_selecionadas = X_scaled.columns[support]
print("Características selecionadas:", features_selecionadas)

plt.figure(figsize=(12, 6))
plt.bar(X_scaled.columns, selector.scores_)
plt.xlabel('Características')
plt.ylabel('Importância (F-score)')
plt.title('Seleção de Características')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
