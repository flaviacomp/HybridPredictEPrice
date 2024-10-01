import pandas as pd
from DataNormalization import DataNormalization
from ForecastModels import ForecastModels
import argparse
import numpy as np
np.random.seed(1234)
#/Users/flaviamonteiro/PycharmProjects/HybridPredictEPrice/data/Dados_Sudeste-Centro-Oeste(Dados SECO).csv
#start='2012-01-01', end='2024-05-25'

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

def calcLimits(df_mpc_proj):
    from scipy.stats import t
    # Calcular a média e o desvio padrão da série temporal
    mean_series = df_mpc_proj['Forecast LPC'].mean()
    std_series = df_mpc_proj['Forecast LPC'].std()

    # Número de observações
    n = len(df_mpc_proj)

    # Grau de liberdade
    degrees_of_freedom = n - 1

    # Valor crítico da distribuição t para 95% de confiança
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_critical = t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Calcular o intervalo de confiança
    margin_of_error = t_critical * (std_series / np.sqrt(n))

    # Limites superior e inferior do intervalo de confiança
    df_mpc_proj['lower_limit'] = df_mpc_proj['Forecast LPC'] - margin_of_error
    df_mpc_proj['upper_limit'] = df_mpc_proj['Forecast LPC'] + margin_of_error

    return df_mpc_proj


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
    end_forecast= '2026-12-31'

    meses = pd.date_range(start=pd.to_datetime(start_forecast) + pd.Timedelta(weeks=1), end=end_db, freq='W')  # "2025-12-31"
    print(meses)

    parser = argparse.ArgumentParser(description="Path, start_db, end_db, start_forecast, end_forecast")
    parser.add_argument("-path", "--num1", type=str, required=False, help="Path")
    parser.add_argument("-start_db", "--num2", type=str, required=False, help="start_db")
    parser.add_argument("-end_db", "--num3", type=str, required=False, help="end_db")
    parser.add_argument("-start_forecast", "--num4", type=str, required=False, help="start_forecast")
    parser.add_argument("-end_forecast", "--num5", type=str, required=False, help="end_forecast")

    args = parser.parse_args()

    df = loadFile(path, start_db, end_db)
    df_norm, scaler_LPC = DataNormalization().apply(df)

    forecast = ForecastModels(start_forecast, end_forecast)

    best_model_residual, pred_residual, metrics = forecast.trainResidual(df_norm, is_graph = True, verbose=False)

    best_model_seasonal, confidence_intervals_seasonal, forecast_mean_seasonal = forecast.predictSeasonal(df_norm, meses, 3, is_graph=True)

    residual_forecast = forecast.predictResidual(df_norm, 3, best_model_residual, forecast_mean_seasonal )

    best_model_sarimax_trend, best_model_lstm_trend, hybrid_trend_forecast = forecast.predictTrend(df_norm,1, 2, is_graph=True, is_test = True)

    result_forecast = forecast.createPredict(df, df_norm,3, scaler_LPC, hybrid_trend_forecast, forecast_mean_seasonal, residual_forecast)

    result_forecast_limits = calcLimits(result_forecast)

    result_forecast_limits.to_csv("resultForecast.csv", sep=';', decimal=',')

    # Calcula o MAPE com intervalo de confiança
    mape_value, confidence_interval = mape_with_confidence_interval(
        df['LPC_SECO'].loc[df.index > start_forecast], result_forecast_limits['Forecast LPC'].loc[result_forecast_limits.index <=end_db], result_forecast_limits['lower_limit'].loc[result_forecast_limits.index <=end_db], result_forecast_limits['upper_limit'].loc[result_forecast_limits.index <=end_db]
    )

    print(f'MAPE: {mape_value:.2f}%')
    print(f'Intervalo de Confiança (95%): ({confidence_interval[0]:.2f}%, {confidence_interval[1]:.2f}%)')


if __name__ == "__main__":
    main()