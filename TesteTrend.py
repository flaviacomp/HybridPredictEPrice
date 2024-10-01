import pandas as pd
from DataNormalization import DataNormalization
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ForecastModels import ForecastModels
import argparse
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

def loadFile(path, start, end):
    ## Carregando o arquivo base
    df = pd.read_csv(path, delimiter=";")
    df.set_index("DATA", inplace=True)
    dates = pd.date_range(start=start, end=end, freq='W')
    print(dates)
    print(df.index)
    df.index = dates
    df_ = df.drop(columns=["CE_SECO", "CMO_SECO","GE_SECO",	"DM_SECO",	"PLD_SECO",'MPI_SECO_50','LPI_SECO_50'])
    return df_

def mape_with_confidence_interval(actual, forecast, lower_limit, upper_limit, alpha=0.05):
    """
    Calcula o MAPE e o intervalo de confiança.

    Args:
      actual: Lista ou array com os valores reais.
      forecast: Lista ou array com os valores previstos.
      lower_limit: Lista ou array com os limites inferiores do intervalo de confiança.
      upper_limit: Lista ou array com os limites superiores do intervalo de confiança.
      alpha: Nível de significância para o intervalo de confiança (padrão 0.05).

    Returns:
      Uma tupla com o MAPE e o intervalo de confiança.
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    lower_limit = np.array(lower_limit)
    upper_limit = np.array(upper_limit)

    # Evita divisão por zero
    actual[actual == 0] = 1e-8

    # Calcula o MAPE
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100

    # Calcula o erro absoluto percentual para cada ponto
    ape = np.abs((actual - forecast) / actual)

    # Calcula a variância do APE
    ape_variance = np.var(ape)

    # Calcula o erro padrão do MAPE
    mape_std_error = np.sqrt(ape_variance / len(actual))

    # Calcula o intervalo de confiança
    z_critical = 1.96  # Para um nível de confiança de 95%
    lower_bound = mape - z_critical * mape_std_error
    upper_bound = mape + z_critical * mape_std_error

    return mape, (lower_bound, upper_bound)


def main():
    path = "/Users/flaviamonteiro/PycharmProjects/HybridPredictEPrice/data/Dados_Sudeste-Centro-Oeste(Dados SECO).csv"
    start_db='2012-01-07'
    end_db='2024-08-18'
    start_forecast='2023-12-31'
    end_forecast= '2033-12-18'

    meses = pd.date_range(start=pd.to_datetime(start_forecast) + pd.Timedelta(weeks=1), end=end_db, freq='W')  # "2025-12-31"

    df = loadFile(path, start_db, end_db)
    df_norm, scaler_LPC = DataNormalization().apply(df)

    print(df_norm)

    # 1ª Parte - Treinar modelo SARIMA
    SARIMA_model_trend = SARIMAX(df_norm['trend'].loc[df_norm.index <= "2024-05-14"].dropna(), order=(2, 1, 2),
                                 seasonal_order=(1, 1, 1, 52), simple_differencing=False)
    SARIMA_model_fit_trend = SARIMA_model_trend.fit(disp=False)
    forecast_result_trend = SARIMA_model_fit_trend.get_forecast(steps=520)
    sarimax_predict = forecast_result_trend.predicted_mean
    confidence_intervals_trend = forecast_result_trend.conf_int()


    sarimax_predict.plot()
    df_norm['trend'].loc[df_norm.index > (pd.to_datetime(start_forecast))].plot()
    df_norm['trend'].plot()
    plt.show()


if __name__ == "__main__":
    main()