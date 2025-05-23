import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    with open('datasets/set3/training.txt') as f:
        dataset_text = json.load(f)
    return dataset_text, f, json


@app.cell
def _(dataset_text):
    import pandas as pd

    dataset = pd.json_normalize(dataset_text, "scripts", max_level=3, errors="ignore")
    dataset1 = dataset.explode(['scripts_generated.rules'])

    dataset1_rules = dataset1['scripts_generated.rules'].apply(pd.Series)
    # dataset1.join(dataset1_rules, how='right')
    # dataset2 = pd.concat([dataset1, dataset1_rules], axis=1)
    dataset2 = dataset1.reset_index()
    dataset2['entry'] = dataset2.groupby(['timestamp', 'index']).cumcount() + 1
    dataset2['scripts_generated.rule_id'] = dataset2['scripts_generated.rules'].apply(lambda x: x.get('rule_id'))
    dataset2['scripts_generated.weight'] = dataset2['scripts_generated.rules'].apply(lambda x: x.get('weight'))
    dataset2_piv = dataset2.pivot_table(
        index=['timestamp'],
        columns='scripts_generated.rule_id',
        values='scripts_generated.weight',
        aggfunc='first'
    )
    dataset2_piv.fillna(inplace=True, method='ffill')
    dataset2_piv.fillna(inplace=True, method='bfill')
    dataset2_piv.columns = [f"rule_{col}" for col in dataset2_piv.columns]
    meta_cols = [
        col for col in dataset2.columns if col not in 
            ['scripts_generated.rules', 'parsed_rule', 'scripts_generated.rule_id', 'scripts_generated.weight']
    ]
    dataset2_metadata = dataset2[meta_cols].drop_duplicates(subset=['timestamp'])
    dataset3 = pd.merge(dataset2_metadata, dataset2_piv, on=['timestamp'])
    return (
        dataset,
        dataset1,
        dataset1_rules,
        dataset2,
        dataset2_metadata,
        dataset2_piv,
        dataset3,
        meta_cols,
        pd,
    )


@app.cell
def _():
    # 
    return


@app.cell
def _(dataset3, mo):
    def split_dataframe(df, reset_column, reset_value):
        """
        Splits a DataFrame into a list of DataFrames based on a reset condition in a column.

        Args:
            df: The input DataFrame.
            reset_column: The name of the column to check for the reset condition.
            reset_value: The value in the reset_column that indicates a reset.

        Returns:
            A list of DataFrames, where each DataFrame represents a segment of the original DataFrame
            starting from a row where the reset_column equals the reset_value.
        """
        split_indices = df[df[reset_column] == reset_value].index.tolist()
    
        # If no reset values are found, return the original DataFrame in a list
        if not split_indices:
            return [df]

        # Ensure the first split index is 0 if it's not already
        if split_indices[0] != 0:
            split_indices = [0] + split_indices

        # Split the DataFrame
        split_dfs = []
        for i in range(len(split_indices) - 1):
            start_index = split_indices[i]
            end_index = split_indices[i+1]
            split_dfs.append(df.iloc[start_index:end_index].copy())  # Use .copy() to avoid SettingWithCopyWarning
    
        # Append the last segment
        split_dfs.append(df.iloc[split_indices[-1]:].copy())

        return split_dfs

    # Split the dataset3 DataFrame based on 'parameters.current_hp' resetting to 200
    split_dataframes = split_dataframe(dataset3, 'parameters.current_hp', 200)

    # Print the number of resulting DataFrames
    mo.md(f"Number of split dataframes: {len(split_dataframes)}")

    # You can now access each DataFrame in the split_dataframes list.
    # For example, to view the first DataFrame:
    split_dataframes[0]
    return split_dataframe, split_dataframes


@app.cell
def _(pd, split_dataframes):
    datasets1 = [*split_dataframes[0:2], pd.concat(split_dataframes[2:7]), split_dataframes[7]]
    return (datasets1,)


@app.cell
def _(datasets1, output_feats):
    datasets1[0][output_feats]
    return


@app.cell
def _(datasets1):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.compose import make_column_transformer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # 1. Feature Selection
    input_feats = [
        'parameters.attacks_landed.lower',
        'parameters.attacks_landed.upper', 
        'parameters.current_hp',
        'parameters.defenses.crouching', 
        'parameters.defenses.standing',
        'parameters.lower_hits', 
        'parameters.upper_hits',
    ]  # Replace with your desired features
    output_feats = [
        "rule_1.0",
        "rule_2.0",
        "rule_3.0",
        "rule_4.0",
        "rule_5.0",
        "rule_6.0",
        "rule_7.0",
        "rule_8.0",
        "rule_9.0",
        "rule_11.0",
        "rule_12.0",
        "rule_13.0",
        "rule_14.0",
    ]

    # 2. Data Normalization/Scaling
    preprocessor = make_column_transformer(
        (MinMaxScaler(), input_feats),
        ('passthrough', output_feats)
    )
    # 3. Sequence Padding/Truncation
    max_sequence_length = 5  # Choose an appropriate length

    processed_data = []
    for df in datasets1:
        # Select features
        feature_data = df[input_feats + output_feats].copy()
        # print(feature_data)
        # print(output_data)

        # Scale the data
        scaled_data = preprocessor.fit_transform(feature_data)
        # print(scaled_data)

        # Pad or truncate sequences
        padded_sequence = pad_sequences(
            [scaled_data], maxlen=max_sequence_length, padding="pre", truncating="pre", dtype='float32'
        )[0]  # pad_sequences expects a list of sequences
        # print(padded_sequence)

        processed_data.append(padded_sequence)

    # 4. Reshape into 3D Array
    processed_data = np.array(processed_data)
    print(f"Shape of processed data: {processed_data.shape}")
    # print(processed_data)
    processed_data
    return (
        MinMaxScaler,
        StandardScaler,
        df,
        feature_data,
        input_feats,
        make_column_transformer,
        max_sequence_length,
        np,
        output_feats,
        pad_sequences,
        padded_sequence,
        preprocessor,
        processed_data,
        scaled_data,
    )


@app.cell
def _(input_feats, max_sequence_length, processed_data):
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed
    from sklearn.model_selection import train_test_split

    input_feats_len = len(input_feats)
    X = processed_data[:, :, :input_feats_len]
    y = processed_data[:, :, input_feats_len:]

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    input_layer = Input(shape=(max_sequence_length, X.shape[2]))
    lstm_layer = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    output_layer = TimeDistributed(Dense(y.shape[2], activation='sigmoid'))(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=5,
        validation_data=(X_test, y_test)
    )

    loss, mse, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    return (
        Dense,
        Input,
        LSTM,
        Model,
        TimeDistributed,
        X,
        X_test,
        X_train,
        history,
        input_feats_len,
        input_layer,
        keras,
        loss,
        lstm_layer,
        mae,
        model,
        mse,
        output_layer,
        train_test_split,
        y,
        y_test,
        y_train,
    )


@app.cell
def _(model):
    model.save('./app/lstm/nds_0_0_3.keras', overwrite=False)
    return


if __name__ == "__main__":
    app.run()
