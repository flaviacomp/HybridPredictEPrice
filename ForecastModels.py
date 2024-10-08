import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.regularizers import l2
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ForecastModels:
    def __init__(self, start_forecast, end_forecast):
        self.models = "Gradient Boost, LSTM, SARIMA"
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast
    def trainResidual(self, df_, is_graph = False, verbose = False):
        np.random.seed(1234)
        print(xgb.__version__) #2.1.1
        # features definidas
        df_normalizado = pd.DataFrame({'LPC_SECO_NORMALIZADA': df_["LPC_SECO_NORMALIZADA"],
                                       'seasonal': df_['seasonal'],
                                       'residual': df_['residual'],
                                       'volatility': df_['residual'].rolling(window=3).std(),
                                       'rolling_mean_5': df_['residual'].shift(1).rolling(window=5).mean(),
                                       'rolling_mean_12': df_['residual'].shift(1).rolling(window=12).mean(),
                                       }, index=df_.index)
        df_normalizado = df_normalizado.dropna()

        # Aplicando o one-hot-vector
        df_normalizado['month'] = df_normalizado.index.month.astype(int)
        df_one_hot = pd.get_dummies(df_normalizado['month'], columns=['month'], prefix='month').astype(int)
        df_normalizado = df_normalizado.drop(columns=['month'])
        df_normalizado = pd.concat([df_normalizado, df_one_hot], axis=1)

        # criando lags features
        num_lags = 9

        for lag in range(1, num_lags + 1):
            df_normalizado[f'lag_{lag + 3}'] = df_normalizado['residual'].shift(lag + 3)
            df_normalizado[f'lag_{lag + 3}'] = df_normalizado[f'lag_{lag + 3}'].bfill()  # prestar atenção aqui

        # Splitting features and target
        X = df_normalizado.drop(columns=['residual', 'LPC_SECO_NORMALIZADA'], axis=1).loc[
            df_normalizado.index < "2023-12-31"] #self.start_forecast
        y = df_normalizado['residual'].loc[df_normalizado.index < "2023-12-31"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=654)

        best_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.8
        )


        """objective = 'reg:squarederror',
        n_estimators = 5000,
        learning_rate = 0.1,
        max_depth = 16,
        subsample = 0.8,
        colsample_bytree = 0.8"""
        # Treinar o modelo ANTIGO
        #history = best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose)
        #NOVO - TESTE REGREDINDO A VERSÃO DO XGBOOST
        def mape_eval(preds, dtrain):
            labels = dtrain.get_label()
            return 'mape', np.mean(np.abs((labels - preds) / (labels + 1))) * 100

        eval_result = {}
        history = best_model.fit(X_train, y_train,  eval_set=[(X_test, y_test)], verbose=True)

        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("MAE:", mae)
        print("RMSE:", rmse)
        #print("MAPE:", history.evals_result_['validation_0']['mape'][0])

        if(is_graph):
            plt.figure(figsize=(10, 6))

            plt.plot([i for i in range(len(y_test))], y_test, label="Teste Real")
            plt.plot([i for i in range(len(y_pred))], y_pred, label="Projetado")
            plt.ylim([-0.5, 0.5])
            plt.legend()
            plt.show()

        return best_model, y_pred, [mae,rmse]

    def predictSeasonal(self, df_, meses, anos_projetados, is_graph = False):
        SARIMA_model_seasonal = SARIMAX(df_['seasonal'].loc[df_.index <= self.start_forecast].dropna(),
                                        order=(1, 0, 1), seasonal_order=(1, 1, 1, 52), simple_differencing=False)
        SARIMA_model_fit_seasonal = SARIMA_model_seasonal.fit(disp=False)
        forecast_result_seasonal = SARIMA_model_fit_seasonal.get_forecast(steps=anos_projetados*52)
        forecast_mean_seasonal = forecast_result_seasonal.predicted_mean
        confidence_intervals_seasonal = forecast_result_seasonal.conf_int()

        # Evaluation
        mse = mean_squared_error(df_['seasonal'][df_.index > self.start_forecast],
                                 forecast_mean_seasonal.iloc[-len(meses):].values.reshape(-1, 1))
        mae = mean_absolute_error(df_['seasonal'][df_.index > self.start_forecast],
                                  forecast_mean_seasonal.iloc[-len(meses):].values.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(df_['seasonal'][df_.index > self.start_forecast],
                                          forecast_mean_seasonal.iloc[-len(meses):].values.reshape(-1, 1)))
        print("MAE:", mae)
        print("RMSE:", rmse)

        if(is_graph):
            forecast_mean_seasonal.plot()
            df_['seasonal'][df_.index > self.start_forecast].plot()
            plt.show()

            forecast_mean_seasonal.iloc[-anos_projetados*52:].plot(label="predict")
            df_['seasonal'][df_.index > self.start_forecast].plot(label="real")
            plt.ylim([-0.4, 0.4])
            plt.show()

        return SARIMA_model_fit_seasonal, confidence_intervals_seasonal, forecast_mean_seasonal

    def predictTrend(self, df_,anos_projetados_sarima, anos_projetados_lstm, is_graph = False, is_test = False):

        # 1ª Parte - Treinar modelo SARIMA
        SARIMA_model_trend = SARIMAX(df_['trend'].loc[df_.index <= self.start_forecast].dropna(), order=(1, 1, 2),#9, 1, 12
                                     seasonal_order=(1, 1, 1, 52), simple_differencing=False)
        SARIMA_model_fit_trend = SARIMA_model_trend.fit(disp=False)
        forecast_result_trend = SARIMA_model_fit_trend.get_forecast(steps=anos_projetados_sarima*52)
        sarimax_predict = forecast_result_trend.predicted_mean
        confidence_intervals_trend = forecast_result_trend.conf_int()


        if(is_graph):
            print("&&&&&&&&&&&&&&&&&& TREND &&&&&&&&&&&&&&&&&&&&&&")
            import matplotlib.pyplot as plt
            sarimax_predict.plot()
            df_['trend'].loc[df_.index > (pd.to_datetime(self.start_forecast))].plot()
            #df_['trend'].plot()
            plt.show()

        # 2ª parte: Treinar modelo LSTM
        # Preparar os dados para o modelo LSTM
        def create_dataset(series, time_step=1):
            data = series.values.reshape(-1, 1)
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                a = data[i:(i + time_step), 0]
                X.append(a)
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        ## usando dois anos da projeção do sarimax como entrada do modelo

        df_concat_sarimax = pd.concat([df_['trend'].loc[df_.index <= self.start_forecast],
                                       sarimax_predict.loc[sarimax_predict.index <= pd.to_datetime(self.start_forecast)+pd.Timedelta(weeks=52*anos_projetados_sarima)]])
        print(df_concat_sarimax.head())
        # Criar dataset
        time_step = 52  # Usando um ano de histórico para prever o próximo valor
        X, Y = create_dataset(df_concat_sarimax, time_step)

        # Redimensionar para LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Dividir em conjunto de treino e teste
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

        # Redimensionar para LSTM [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        if(is_test == False):
            # Construir o modelo LSTM
            model = Sequential()
            model.add(
                Bidirectional(LSTM(200, return_sequences=False, kernel_regularizer=l2(0.01)), input_shape=(time_step, 1)))
            model.add(Dense(1, kernel_regularizer=l2(0.01)))

            # Compilar o modelo
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Treinar o modelo
            model.fit(X_train, Y_train, batch_size=32, epochs=500)#500

            model.save('meu_modelo_tf.h5')
        else:
            from tensorflow.keras.models import load_model

            # Carregue o modelo
            model = load_model('meu_modelo_tf.h5')

        # Previsão e projeção da tendência
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Prever os próximos 10 anos (120 meses)
        future_steps = 52 * anos_projetados_lstm
        last_values = X_test[-1]

        lstm_predict = []
        for _ in range(future_steps):
            prediction = model.predict(last_values.reshape(1, time_step, 1))
            lstm_predict.append(prediction[0, 0])
            last_values = np.append(last_values[1:], prediction)

        # Exibir os resultados
        import matplotlib.pyplot as plt
        # Exibir os resultados
        #plt.figure(figsize=(15, 6))
        #plt.plot(range(len(df_concat_sarimax)), df_concat_sarimax, label='Original Residual')
        #plt.plot(range(len(df_concat_sarimax), len(df_concat_sarimax) + len(lstm_predict)), lstm_predict,
        #         label='Future Residual')
        #plt.legend()
        #plt.show()

        # Avaliação do modelo,
        print("Treino:")
        print("mean_squared_error: ", mean_squared_error(Y_train, train_predict))
        print("mean_absolute_error: ", mean_absolute_error(Y_train, train_predict))
        print("Teste:")
        print("mean_squared_error: ", mean_squared_error(Y_test, test_predict))
        print("mean_absolute_error: ", mean_absolute_error(Y_test, test_predict))

        hybrid_forecast = self.hybridTrend(lstm_predict, anos_projetados_sarima, sarimax_predict)

        return SARIMA_model_fit_trend, model, hybrid_forecast


    def hybridTrend(self, lstm_predict,anos_projetados_sarimax,  sarimax_predict):
        from scipy.interpolate import interp1d

        meses = pd.date_range(start=pd.to_datetime(self.start_forecast)+pd.Timedelta(weeks=(anos_projetados_sarimax*52)+1), end=self.end_forecast, freq='W')#"2025-12-31"
        #print("##########################")
        print(len(meses), len(lstm_predict))
        print(meses)
        print(lstm_predict.index)
        #print("##########################")
        lstm_mean_trend = pd.DataFrame(lstm_predict, columns=['predicted_mean'])
        lstm_mean_trend.index = meses

        # Definir o índice de data
        #sarimax_mean_trend = pd.DataFrame(sarimax_predict)
        #sarimax_mean_trend.set_index(sarimax_mean_trend.index, inplace=True)

        #novo jeito
        sarimax_mean_trend = pd.DataFrame(sarimax_predict.loc[sarimax_predict.index < pd.to_datetime(self.start_forecast)+pd.Timedelta(weeks=(anos_projetados_sarimax*52)+1)])
        sarimax_mean_trend.set_index(sarimax_mean_trend.index, inplace=True)

        # Definindo o número de pontos para a transição suave
        transition_points = 10
        # Parte final da primeira série e início da segunda
        end_of_original = sarimax_mean_trend['predicted_mean'].iloc[-int(transition_points):].values
        start_of_future = lstm_mean_trend['predicted_mean'].iloc[:int(transition_points)].values

        # Interpolação manual usando média ponderada entre o fim da série original e o início da futura
        #blended_section = (1 - np.linspace(0, 1, transition_points)) * end_of_original + np.linspace(0, 1,
        #                                                                                          transition_points) * start_of_future
        # Aplicando a função sigmoide para suavização mais gradual
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        blend_weights = sigmoid(np.linspace(-20, 20, transition_points))  # Transição suave não-linear
        blended_section = (1 - blend_weights) * end_of_original + blend_weights * start_of_future

        # Criando o DataFrame para a seção interpolada
        blended_index = pd.concat([sarimax_mean_trend.iloc[-int(transition_points/2):],lstm_mean_trend.iloc[:int(transition_points/2)]]).index # Usando as datas do final da primeira série

        blended_section_df = pd.DataFrame(blended_section, index=blended_index, columns=['predicted_mean'])

        # Série fundida
        hybrid_forecast = pd.concat(
            [sarimax_mean_trend.iloc[:-int(transition_points/2)], blended_section_df, lstm_mean_trend[int(transition_points/2):]])


        #hybrid_forecast_2 = pd.concat([hybrid_forecast_2, lstm_mean_trend.iloc[transition_points:]])

        #hybrid_forecast = pd.concat([sarimax_mean_trend.loc[sarimax_mean_trend.index < pd.to_datetime(self.start_forecast)+pd.Timedelta(weeks=105)], lstm_mean_trend])

        #hybrid_forecast.plot()

        return hybrid_forecast


    def predictResidual(self, df_normalizado, anos_projetados, best_model, forecast_mean_seasonal):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # criando lags features
        num_lags = 9
        # Função para fazer previsões iterativas
        def iterative_prediction(model, df, target, steps, data_inicial):
            # Cria uma cópia do dataframe para evitar modificar o original
            df_pred = df.copy()

            for step in range(steps):

                data_inicial = data_inicial + pd.DateOffset(weeks=1)

                # Convertendo o mês em one-hot encoding
                df_temp = pd.DataFrame({'month': [data_inicial.month]})
                df_one_hot = pd.get_dummies(df_temp, columns=['month'], prefix='month').astype(int)

                for i in range(1, 13):
                    if f'month_{i}' not in df_one_hot.columns:
                        df_one_hot[f'month_{i}'] = 0

                new_data = {
                    'seasonal': forecast_mean_seasonal[step],
                    'volatility': df_pred['residual'].rolling(window=3).std().iloc[-1],
                    'rolling_mean_5': df_pred['residual'].rolling(window=5).mean().iloc[-1],
                    'rolling_mean_12': df_pred['residual'].rolling(window=12).mean().iloc[-1],
                    'month_1': df_one_hot['month_1'].values[0],
                    'month_2': df_one_hot['month_2'].values[0],
                    'month_3': df_one_hot['month_3'].values[0],
                    'month_4': df_one_hot['month_4'].values[0],
                    'month_5': df_one_hot['month_5'].values[0],
                    'month_6': df_one_hot['month_6'].values[0],
                    'month_7': df_one_hot['month_7'].values[0],
                    'month_8': df_one_hot['month_8'].values[0],
                    'month_9': df_one_hot['month_9'].values[0],
                    'month_10': df_one_hot['month_10'].values[0],
                    'month_11': df_one_hot['month_11'].values[0],
                    'month_12': df_one_hot['month_12'].values[0],
                }

                new_df = pd.DataFrame(new_data, index=[0])

                for lag in range(1, num_lags + 1):
                    new_df[f'lag_{lag + 3}'] = df_pred['residual'].iloc[-(lag + 3)]
                    new_df[f'lag_{lag + 3}'] = new_df[f'lag_{lag + 3}'].bfill()

                new_pred = model.predict(new_df)

                new_row = {
                    'residual': new_pred[0],
                    'seasonal': new_data['seasonal'],
                    'volatility': new_data['volatility'],
                    'rolling_mean_5': new_data['rolling_mean_5'],
                    'rolling_mean_12': new_data['rolling_mean_12'],
                    'month_1': new_data['month_1'],
                    'month_2': new_data['month_2'],
                    'month_3': new_data['month_3'],
                    'month_4': new_data['month_4'],
                    'month_5': new_data['month_5'],
                    'month_6': new_data['month_6'],
                    'month_7': new_data['month_7'],
                    'month_8': new_data['month_8'],
                    'month_9': new_data['month_9'],
                    'month_10': new_data['month_10'],
                    'month_11': new_data['month_11'],
                    'month_12': new_data['month_12'],
                }

                new_row = pd.DataFrame(new_row, index=[0])

                for lag in range(1, num_lags + 1):
                    new_row[f'lag_{lag + 3}'] = new_df[f'lag_{lag + 3}']

                new_row.index = [data_inicial]

                df_pred = pd.concat([df_pred, new_row], ignore_index=False)

                print(data_inicial)
            return df_pred

        # Número de passos de previsão (ex. 4 semanas por mês)
        steps = anos_projetados * 52
        data_inicial = pd.Timestamp("2024-01-31")

        # Usando os dados mais recentes como ponto de partida para as previsões
        df_predictions = iterative_prediction(best_model, df_normalizado.loc[(df_normalizado.index < "2024-01-31")], ###AQUI
                                              "residual", steps, data_inicial)

        return df_predictions

    def createPredict(self, df_, df_norm, anos_projetados, scaler_LPC, hybrid_trend_forecast, seasonal_forecast, residual_forecast):
        meses = pd.date_range(start=pd.to_datetime(self.start_forecast) + pd.Timedelta(weeks=1), end=pd.to_datetime(self.end_forecast) , freq='W') #- pd.Timedelta(weeks=1)

        df_hybrid_trend_forecast = pd.DataFrame(hybrid_trend_forecast.loc[hybrid_trend_forecast.index <= pd.to_datetime(self.end_forecast) ].values,
                                          columns=["Resultado"])
        #print("####hybrid_trend_forecast####")
        #for data in hybrid_trend_forecast.index:
        #    print(data)
        #print("####### meses #######")
        #print(meses)
        df_hybrid_trend_forecast.index = meses
        #print("#############################")
        df_forecast_mean_seasonal = pd.DataFrame(seasonal_forecast.values, columns=["Resultado"])
        df_forecast_mean_seasonal.index = meses

        df_predictions_residual = pd.DataFrame(residual_forecast['residual'].iloc[-anos_projetados*52:].values, columns=["Resultado"])
        df_predictions_residual.index = meses

        #lpc_pred = df_predictions_residual['Resultado'] + df_hybrid_trend_forecast['Resultado'] + df_forecast_mean_seasonal[
        #    'Resultado']
        lpc_pred = df_norm['trend'].loc[df_norm.index >= pd.to_datetime(self.start_forecast) + pd.Timedelta(weeks=1)].iloc[:30] + df_predictions_residual['Resultado'].iloc[:30] + df_norm['seasonal'].loc[df_norm.index >= pd.to_datetime(self.start_forecast) + pd.Timedelta(weeks=1)].iloc[:30]

        lpc_pred = scaler_LPC.inverse_transform(lpc_pred.values.reshape(-1, 1))

        df_lpc_proj = pd.DataFrame(lpc_pred.flatten(), columns=['Forecast LPC'])
        df_lpc_proj.index = meses[:30]

        df_lpc_proj.plot()
        df_['LPC_SECO'].loc[df_.index >= self.start_forecast].plot(label="Real LPC")###AQUI
        # plt.ylim([50,250])

        #df_norm['residual'].loc[df_norm.index >= pd.to_datetime(self.start_forecast) + pd.Timedelta(weeks=1)].iloc[:30].plot(label="Real")
        #df_predictions_residual['Resultado'].iloc[:30].plot(label="Predict")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show()

        return df_lpc_proj