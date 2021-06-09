# %%
import time
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.regularizers import L1L2

master_df = pd.read_csv('Daily_data/master_df_binary.csv', index_col=[0], parse_dates=True)
master_df_only_telecom = master_df[['High_vz', 'Low_vz', 'Open_vz', 'Close_vz', 'Volume_vz', 'Adj Close_vz',
                                    'High_atnt', 'Low_atnt', 'Open_atnt', 'Close_atnt', 'Volume_atnt', 'Adj Close_atnt',
                                    'High_nippon', 'Low_nippon', 'Open_nippon', 'Close_nippon', 'Volume_nippon', 'Adj Close_nippon',
                                    'High_softbank', 'Low_softbank', 'Open_softbank', 'Close_softbank', 'Volume_softbank', 'Adj Close_softbank',
                                    'High_deutsche', 'Low_deutsche', 'Open_deutsche', 'Close_deutsche', 'Volume_deutsche', 'Adj Close_deutsche',
                                    'High_kpn', 'Low_kpn', 'Open_kpn', 'Close_kpn', 'Volume_kpn', 'Adj Close_kpn', 'verizon_binary']].copy()

master_df_only_close = master_df[['Close_vz',
                       'Close_atnt',
                       'Close_nippon',
                       'Close_softbank',
                       'Close_deutsche',
                       'Close_kpn',
                       'Close_sp500', 'Close_dow', 'Close_russell', 'Close_nasdaq', 'Close_aex', 'Close_dax', 'Close_nikkei',
                       # 'Close_usd_yen','Close_usd_eur',
                       'Close_gold', 'Close_silver', 'Close_oil',
                       # 'Close_platinum'
                       'Close_10ybond', 'Close_vix', 'Close_dixie', 'verizon_binary']].copy()
master_df_only_close.fillna(method='bfill', inplace=True)
master_df_only_close.dropna(inplace=True)

train_df_year_telecom = master_df_only_telecom['2000':'2017'].copy()
val_df_year_telecom = master_df_only_telecom.loc['2018'].copy()
train_val_df_year_telecom = master_df_only_telecom['2000':'2018'].copy()
test_df_year_telecom = master_df_only_telecom.loc['2019'].copy()

train_df_year_close = master_df_only_close['2000':'2017'].copy()
val_df_year_close = master_df_only_close.loc['2018'].copy()
train_val_df_year_close = master_df_only_close['2000':'2018'].copy()
test_df_year_close = master_df_only_close.loc['2019'].copy()
SEQUENCE_LENGTH = 31


def create_sequences(df_param):
    sequential_data = []
    prev_days = deque(maxlen=SEQUENCE_LENGTH)  # Create deque object
    for i in df_param.values:  # Loop over rows in np array
        prev_days.append([n for n in i[:-1]])  # Append rows of data to deque object
        if len(prev_days) == SEQUENCE_LENGTH:  # Untill deque reaches sequence length
            sequential_data.append([np.array(prev_days), i[
                -1]])  # When that is the case, append the deque object as np array together with target
    X = []
    y = []
    for sequence, target in sequential_data:
        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)

# regularizers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

def sentdex_model(X_train):
    model = Sequential()
    model.add(LSTM(33, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(LSTM(33, input_shape=(X_train.shape[1:])))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.optimizers.Adam()
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    # model.summary()
    return model


# scaler = MinMaxScaler(feature_range=(-1, 1))
# train_df.iloc[:, :-1] = scaler.fit_transform(train_df.iloc[:, :-1].values)
# X_train, y_train = create_sequences(train_df)
# sent_mod = sentdex_model(X_train)
# sent_mod.fit(X_train, y_train, epochs=35, batch_size=10, verbose=2)


def rolling_forecast(train_df, val_or_test_df):
    # First, we stack the dataframes so we can easily index the right train and validation data in the loop.
    dataframes_stacked = pd.concat([train_df, val_or_test_df])
    date_times = []
    model_predictions = []
    true_values = []
    for i in range(0, len(val_or_test_df)):
        loop_start = time.time()
        # Slice the proper size of the train_val_df as train data. Make a copy to make sure dataframe
        # is not modified in place.
        train_df_loop = dataframes_stacked.iloc[:len(train_df) + i].copy()
        # Slice the proper size of the train_val_df as validation data. Make a copy to make sure dataframe
        # is not modified in place.
        val_series_loop = dataframes_stacked.iloc[len(train_df) + i:len(train_df) + 1 + i].copy()
        # Fit-transform scaler on X_train data, transform on X_val_or_test data. Do this to avoid data leakage.
        # Only fit-transform predictors, so leave target untouched.
        # Here is also the place where additional feature engineering shall be applied
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df_loop.iloc[:, :-1] = scaler.fit_transform(train_df_loop.iloc[:, :-1].values)
        val_series_loop.iloc[:, :-1] = scaler.transform(val_series_loop.iloc[:, :-1].values)
        # # Stack train and validation data so we can create proper sequence data
        # If we don't do this, the val_series object cannot grab previous timesteps
        stacked_transformed_df = pd.concat((train_df_loop, val_series_loop))
        # Create sequences
        X_sequenced, y_sequenced = create_sequences(stacked_transformed_df)
        # Split data gain into train and val/test. Everything up until last row = train data, last row = val/test data.
        X_train, X_val_or_test, y_train, y_val_or_test = X_sequenced[:-1], X_sequenced[-1], y_sequenced[:-1], \
                                                         y_sequenced[-1]
        # Reshape X_val_or_test so it's in correct shape as it's only 1 row and NN expects 3D ndarray
        X_val_or_test = X_val_or_test.reshape(1, X_val_or_test.shape[0], X_val_or_test.shape[1])
        # Initialize & Fit the model
        model = sentdex_model(X_train)
        history_e1d1 = model.fit(X_train, y_train, epochs=500, batch_size=10, verbose=2)
        # Make predictions with the fitted model
        model_prediction = model.predict(X_val_or_test).item()
        # Generate timestamp & true value data
        val_series_timestamp = val_series_loop.index
        val_series_timestamp_string = val_series_timestamp[0].strftime('%Y-%m-%d')
        true_value = val_series_loop['verizon_binary'].squeeze()
        # Append all data to list
        date_times.append(val_series_timestamp[0])
        model_predictions.append(model_prediction)
        true_values.append(true_value)
        loop_end = time.time()
        run_time = loop_end - loop_start
        print(f"Finished validating timestep {val_series_timestamp_string}. "
              f"This took {round(run_time)} seconds. Model predicted {model_prediction} actual value was {true_value}")
    return date_times, model_predictions, true_values


def generate_results(train_df_param, val_or_test_df, result_type_string):
    date_times, model_predictions, true_values = rolling_forecast(train_df=train_df_param, val_or_test_df=val_or_test_df)
    predictions_df = pd.DataFrame(
        columns=['DateTime', 'model_prediction', 'smoothed_prediction', 'true_value'])
    predictions_df['DateTime'] = date_times
    predictions_df['model_prediction'] = model_predictions
    predictions_df['smoothed_prediction'] = predictions_df['model_prediction'].apply(lambda x: 1 if x > 0.5 else 0)
    predictions_df['true_value'] = true_values
    predictions_df.set_index('DateTime', inplace=True, drop=True)
    predictions_df.to_csv(f'Results/R4_LSTM/Second_experiment/{result_type_string}.csv')

generate_results(train_df_year_close, val_df_year_close, 'val_1_year_close_features500epochs')
generate_results(train_df_year_telecom, val_df_year_telecom, 'val_1year_telecom_features500epochs')
