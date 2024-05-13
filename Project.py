import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# Leitura do arquivo CSV
df = pd.read_csv('C:\\Users\\dados_srag2023.csv')

# Conversão da coluna 'data' para o formato de data
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y').dt.date

# Ordenação do DataFrame por data
df = df.sort_values(by='data')

# Cálculo de estatísticas descritivas
media_casos = np.mean(df['casos'])
mediana_casos = np.median(df['casos'])
desvio_padrao_casos = np.std(df['casos'])
variancia_casos = statistics.variance(df['casos'])

# Plotagem da evolução temporal
plt.figure(figsize=(10, 6))
plt.plot(df['data'], df['casos'], label='Casos Diários', marker='o')
plt.title('Evolução Temporal de Casos de SRAG')
plt.xlabel('Data')
plt.ylabel('Casos Diários')
plt.legend()
plt.grid(True)
plt.show()

# Regressão linear para projeção
x_values = np.arange(len(df)).reshape(-1, 1)
y_values = df['casos'].values.reshape(-1, 1)
regressor = LinearRegression()
regressor.fit(x_values, y_values)

# Coeficientes da regressão
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]

# Projeção para os próximos meses
proximo_mes = len(df) + 1
proximo_mes_projecao = regressor.predict([[proximo_mes]])

print(f"Estatísticas Descritivas:\nMédia: {media_casos}\nMediana: {mediana_casos}\nDesvio Padrão: {desvio_padrao_casos}\nVariância: {variancia_casos}")
print(f"Coeficiente de Inclinação (Slope): {slope}")
print(f"Coeficiente de Interceptação (Intercept): {intercept}")
print(f"Projeção para o próximo mês: {proximo_mes_projecao[0][0]} casos")
result = stats.describe(df['casos'], ddof=1, bias=False)
print(result)
