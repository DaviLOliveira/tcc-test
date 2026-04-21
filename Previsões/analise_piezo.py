
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# 1. CARREGAR DADOS

arquivo = r"C:\Users\brenda.santos\Desktop\TCC 1\Previsões\E1PZ056.csv"

df = pd.read_csv(arquivo)

df.columns = df.columns.str.strip()
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")
df.set_index("datetime", inplace=True)

# 2. TRANSFORMAR PARA SÉRIE MENSAL

serie_mensal = df["valor"].resample("ME").mean()
serie_mensal = serie_mensal.interpolate(method="time")

print("Período analisado:")
print("Início:", serie_mensal.index.min())
print("Fim:", serie_mensal.index.max())

# 3. MODELOS PARA TESTE

modelos = {
    "M1": ((1,1,1),(1,1,1,12)),
    "M2": ((1,1,0),(1,1,0,12)),
    "M3": ((0,1,1),(0,1,1,12)),
    "M4": ((1,1,0),(0,1,1,12)),
}

resultados = {}

print("\n===== TESTANDO MODELOS =====\n")

for nome, config in modelos.items():

    ordem, sazonal = config

    modelo = SARIMAX(
        serie_mensal,
        order=ordem,
        seasonal_order=sazonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    resultado = modelo.fit(disp=False)

    resultados[nome] = {
        "modelo": resultado,
        "AIC": resultado.aic
    }

    print(f"{nome} → AIC: {resultado.aic:.4f}")

# 4. SELECIONAR MELHOR MODELO

melhor_modelo_nome = min(resultados, key=lambda x: resultados[x]["AIC"])
melhor_modelo = resultados[melhor_modelo_nome]["modelo"]

print("\n===== MELHOR MODELO =====")
print("Modelo selecionado:", melhor_modelo_nome)
print("AIC:", resultados[melhor_modelo_nome]["AIC"])
print("\nResumo estatístico:\n")
print(melhor_modelo.summary())

# 5. PREVISÃO FUTURA (12 MESES)

passos = 12

previsao = melhor_modelo.get_forecast(steps=passos)

media_prevista = previsao.predicted_mean
intervalo = previsao.conf_int()

ultima_data = serie_mensal.index[-1]

datas_futuras = pd.date_range(
    start=ultima_data + pd.offsets.MonthEnd(),
    periods=passos,
    freq="ME"
)

df_previsao = pd.DataFrame({
    "data": datas_futuras,
    "previsao": media_prevista.values,
    "limite_inferior": intervalo.iloc[:, 0].values,
    "limite_superior": intervalo.iloc[:, 1].values
})

# 6. SALVAR CSV

caminho_saida = r"C:\Users\brenda.santos\Desktop\TCC 1\Previsões\previsao_sarima_E1PZ056.csv"
df_previsao.to_csv(caminho_saida, index=False)

print("\nArquivo salvo em:", caminho_saida)

# 7. GRÁFICO

plt.figure(figsize=(14,8))

# Série histórica
plt.plot(
    serie_mensal.index,
    serie_mensal,
    label="Série Histórica",
    linewidth=2,
    color="steelblue"
)

# Previsão
plt.plot(
    df_previsao["data"],
    df_previsao["previsao"],
    linewidth=3,
    color="darkorange",
    label="Previsão (12 meses)"
)

# Intervalo de confiança
plt.fill_between(
    df_previsao["data"],
    df_previsao["limite_inferior"],
    df_previsao["limite_superior"],
    alpha=0.20,
    color="orange",
    label="Intervalo de Confiança (95%)"
)

# Títulos
plt.title(
    f"Previsão do Nível Piezométrico – Modelo SARIMA ({melhor_modelo_nome})",
    fontsize=20
)

plt.xlabel("Data", fontsize=22)
plt.ylabel("Nível Piezométrico (m)", fontsize=22)

# Ajuste das datas
ax = plt.gca()

# limite do eixo até outubro de 2026
plt.xlim(serie_mensal.index.min(), pd.Timestamp("2026-10-31"))

# meses de 5 em 5
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))

plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)

# Grade leve 
plt.grid(
    True,
    linestyle="--",
    alpha=0.4
)

# Legenda
plt.legend(
    fontsize=20,
    frameon=True
)

plt.tight_layout()
plt.show()

# 8. ERRO IN-SAMPLE (SEM PERÍODO DE AQUECIMENTO)

pred = melhor_modelo.get_prediction()

previsao_in_sample = pred.predicted_mean

# remover primeiros 24 meses
previsao_estavel = previsao_in_sample.iloc[24:]
serie_estavel = serie_mensal.iloc[24:]

# garantir alinhamento
previsao_estavel = previsao_estavel.loc[serie_estavel.index]

mae = mean_absolute_error(serie_estavel, previsao_estavel)
rmse = np.sqrt(mean_squared_error(serie_estavel, previsao_estavel))

print("\n===== MÉTRICAS IN-SAMPLE (PERÍODO ESTÁVEL) =====")
print(f"MAE: {mae:.4f} m")
print(f"RMSE: {rmse:.4f} m")
