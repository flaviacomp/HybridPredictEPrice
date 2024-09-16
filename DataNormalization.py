from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.seasonal import STL

class DataNormalization:
    def __init__(self):
        self.type = "STL Statsmodels"

    def apply(self, df_):
        #Normalizando a série toda antes da etapa de decomposição da série
        scaler_LPC = MinMaxScaler(feature_range=(1, 2))
        df_['LPC_SECO_NORMALIZADA'] = scaler_LPC.fit_transform(df_['LPC_SECO'].values.reshape(-1, 1))

        #Decompondo a série em Sazonalidade, Tendência e Resíduo. O Resíduo será nossa variável alvo.
        stl = STL(df_['LPC_SECO_NORMALIZADA'], seasonal=53, period=52)
        result = stl.fit()

        # Feature Engineering
        df_['trend'] = result.trend
        df_['seasonal'] = result.seasonal
        df_['residual'] = result.resid

        #features definidas
        df_normalizado = pd.DataFrame({'LPC_SECO_NORMALIZADA': df_["LPC_SECO_NORMALIZADA"],
                                       'seasonal':df_['seasonal'],
                                       'residual':df_['residual'],
                                       'trend': df_['trend'],
                                       'volatility' : df_['residual'].rolling(window=3).std(),
                                       'rolling_mean_5': df_['residual'].shift(1).rolling(window=5).mean(),
                                       'rolling_mean_12': df_['residual'].shift(1).rolling(window=12).mean(),
                                       }, index=df_.index)
        df_normalizado = df_normalizado.dropna()

        #Aplicando o one-hot-vector
        df_normalizado['month'] = df_normalizado.index.month.astype(int)
        df_one_hot = pd.get_dummies(df_normalizado['month'], columns=['month'], prefix='month').astype(int)
        df_normalizado = df_normalizado.drop(columns=['month'])
        df_normalizado = pd.concat([df_normalizado, df_one_hot], axis=1)

        #criando lags features
        num_lags = 9

        for lag in range(1, num_lags+1):
            df_normalizado[f'lag_{lag+3}'] = df_normalizado['residual'].shift(lag+3)
            df_normalizado[f'lag_{lag+3}'] = df_normalizado[f'lag_{lag+3}'].bfill() #prestar atenção aqui

        return df_normalizado, scaler_LPC