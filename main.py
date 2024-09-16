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
    df = pd.read_csv(path, delimiter=",")
    df.set_index("DATA", inplace=True)
    dates = pd.date_range(start=start, end=end, freq='W')
    df.index = dates
    df_ = df.drop(columns=["CE_SECO", "CMO_SECO","GE_SECO",	"DM_SECO",	"PLD_SECO",'MPI_SECO_50','LPI_SECO_50',"Unnamed: 11"])
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

def main():
    path = "/Users/flaviamonteiro/PycharmProjects/HybridPredictEPrice/data/Dados_Sudeste-Centro-Oeste(Dados SECO).csv"
    start_db='2012-01-01'
    end_db='2024-05-25'
    start_validation='2023-12-31'

    parser = argparse.ArgumentParser(description="Path, start, end")
    parser.add_argument("-path", "--num1", type=str, required=False, help="Path")
    parser.add_argument("-start_db", "--num2", type=str, required=False, help="Start")
    parser.add_argument("-end_db", "--num3", type=str, required=False, help="End")

    args = parser.parse_args()

    df = loadFile(path, start_db, end_db)
    df_norm, scaler_LPC = DataNormalization().apply(df)
    print(df_norm['trend'])
    best_model_residual, pred_residual, metrics = ForecastModels().trainResidual(df_norm, is_graph = True, verbose=False)

    best_model_seasonal, confidence_intervals_seasonal, forecast_mean_seasonal = ForecastModels().predictSeasonal(df_norm, is_graph=True)

    residual_forecast = ForecastModels().predictResidual(df_norm, best_model_residual, forecast_mean_seasonal )
    print(residual_forecast['residual'])

    
    best_model_sarimax_trend, best_model_lstm_trend, hybrid_trend_forecast = ForecastModels().predictTrend(df_norm, is_graph=True, is_test = True)
    print(hybrid_trend_forecast)
    result_forecast = ForecastModels().createPredict(df, scaler_LPC, hybrid_trend_forecast, forecast_mean_seasonal, residual_forecast)

    result_forecast_limits = calcLimits(result_forecast)

    result_forecast_limits.to_csv("resultForecast.csv", sep=';', decimal=',')


if __name__ == "__main__":
    main()