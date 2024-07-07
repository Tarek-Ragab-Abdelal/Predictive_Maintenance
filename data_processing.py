import pandas as pd
import numpy as np

def process_data(df):

    # data preprocessing
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")

    # Define telemetry fields
    telemetry_fields = ['volt', 'rotate', 'pressure', 'vibration']

    # Calculate mean values for telemetry features
    telemetry_mean_3h = calculate_telemetry_features(df, telemetry_fields, 'mean', '3h')
    telemetry_sd_3h = calculate_telemetry_features(df, telemetry_fields, 'std', '3h')
    telemetry_mean_24h = calculate_rolling_telemetry_features(df, telemetry_fields, 'mean', '24h', '3h')
    telemetry_sd_24h = calculate_rolling_telemetry_features(df, telemetry_fields, 'std', '24h', '3h')

    telemetry_feat = merge_telemetry_features([telemetry_mean_3h, telemetry_sd_3h, telemetry_mean_24h, telemetry_sd_24h])

    # Process errors
    error_count = process_errors(df)

    # Process components
    comp_rep_df = process_components(df)

    # Process age
    age = process_age(df)

    # Process models
    model_only_df = process_models(df)

    # Final data stream
    input_data = telemetry_feat.merge(error_count, on=['datetime', 'machineID'])
    input_data = pd.concat([input_data, comp_rep_df, age, model_only_df], axis=1)
    input_data = input_data.drop(columns=['datetime', 'machineID'])
    input_data = input_data.astype(np.float32)
    input_data = input_data.fillna(0.0)
    input_data = input_data.values.tolist()

    # Reshape the input data to match the model's expected input shape (samples, features)
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    return input_data_reshaped


def calculate_telemetry_features(df, fields, agg_func, resample_period):
    temp = []
    for col in fields:
        resampled = pd.pivot_table(df, index='datetime', columns='machineID', values=col).resample(resample_period, closed='left', label='right').agg(agg_func).unstack()
        temp.append(resampled)
    telemetry = pd.concat(temp, axis=1)
    telemetry.columns = [f"{i}{agg_func}_{resample_period}" for i in fields]
    telemetry.reset_index(inplace=True)
    return telemetry


def calculate_rolling_telemetry_features(df, fields, agg_func, window, resample_period):
    temp = []
    for col in fields:
        rolling = (pd.pivot_table(df, index='datetime', columns='machineID', values=col)
                   .rolling(window=window)
                   .agg(agg_func)
                   .resample(resample_period, closed='left', label='right')
                   .first()
                   .unstack())
        temp.append(rolling)
    telemetry = pd.concat(temp, axis=1)
    telemetry.columns = [f"{i}{agg_func}_{window}" for i in fields]
    telemetry.reset_index(inplace=True)
    telemetry = telemetry.dropna(subset=[f"{fields[0]}{agg_func}_{window}"])
    return telemetry


def merge_telemetry_features(feature_dfs):
    merged = feature_dfs[0]
    for feature_df in feature_dfs[1:]:
        merged = merged.merge(feature_df, on=['datetime', 'machineID'])
    return merged


def process_errors(df):
    error_count = df.iloc[:, [0, 1]]
    errors_only = df.iloc[:, [6]]
    errors_only_encoded = one_hot_encode(errors_only, ['error1', 'error2', 'error3', 'error4', 'error5'], 'errorID')

    error_count = pd.concat([error_count, errors_only_encoded], axis=1)
    temp = []
    fields = ['error%d' % i for i in range(1, 6)]
    for col in fields:
        rolling_sum = pd.pivot_table(error_count, index='datetime', columns='machineID', values=col).rolling(window=24).sum().resample('3h', closed='left', label='right').first().unstack()
        temp.append(rolling_sum)
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace=True)
    return error_count


def process_components(df):
    comp_rep = df.iloc[:, [0, 1]]
    comp_only = df.iloc[:, [7]]
    comp_rep_only_encoded = one_hot_encode(comp_only, ['comp1', 'comp2', 'comp3', 'comp4'], 'comp')

    comp_rep = pd.concat([comp_rep, comp_rep_only_encoded], axis=1)
    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[comp_rep[comp].notnull(), comp] = comp_rep.loc[comp_rep[comp].notnull(), 'datetime']

    # Ensure the datetime column is of dtype datetime64[ns]
    comp_rep['datetime'] = pd.to_datetime(comp_rep['datetime'])

    for comp in components:
        comp_rep[comp] = pd.to_datetime(comp_rep[comp], errors='coerce')
        comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')

    comp_rep_df = comp_rep.iloc[-1, 2:].to_frame().T
    return comp_rep_df


def process_age(df):
    df['age'] = df['age'].astype(int)
    current_year = pd.to_datetime('today').year
    df['age'] = current_year - df['age']
    age = pd.DataFrame({'age': [df.iloc[-1, 9]]})
    return age


def process_models(df):
    model_only = df.iloc[:, [8]]
    model_only_encoded = one_hot_encode(model_only, ['model1', 'model2', 'model3', 'model4'], 'model')
    model_only_encoded = model_only_encoded.iloc[-1, :].to_frame().T
    return model_only_encoded


def one_hot_encode(data, unique_values, column_name):
    if data.empty:
        return pd.DataFrame([[0] * len(unique_values)], columns=[f"{column_name}_{val}" for val in unique_values])

    df = pd.DataFrame(data, columns=[column_name])
    one_hot_encoded = pd.DataFrame(0, index=df.index, columns=unique_values)
    for i, value in enumerate(df[column_name]):
        if value in unique_values:
            one_hot_encoded.at[i, value] = 1
    return one_hot_encoded

