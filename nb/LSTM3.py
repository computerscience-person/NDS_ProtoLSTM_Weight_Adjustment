import marimo

__generated_with = "0.13.11"
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
    return (dataset_text,)


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
    return dataset3, pd


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
    return (split_dataframes,)


@app.cell
def _(pd, split_dataframes):
    datasets1 = [*split_dataframes[0:2], pd.concat(split_dataframes[2:7]), split_dataframes[7]]
    return (datasets1,)


@app.cell
def _(datasets1):
    datasets1
    return


@app.cell
def _(datasets1):
    import tensorflow as tf
    import numpy as np
    from keras.saving import register_keras_serializable
    from sklearn.model_selection import train_test_split

    # 1. Define input and output feature names
    input_feats = [
        'parameters.attacks_landed.lower',
        'parameters.attacks_landed.upper', 
        'parameters.current_hp',
        'parameters.defenses.crouching', 
        'parameters.defenses.standing',
        'parameters.lower_hits', 
        'parameters.upper_hits',
    ]
    output_feats = [
        "rule_1.0", "rule_2.0", "rule_3.0", "rule_4.0", "rule_5.0",
        "rule_6.0", "rule_7.0", "rule_8.0", "rule_9.0", "rule_11.0",
        "rule_12.0", "rule_13.0", "rule_14.0",
    ]

    # 2. Prepare raw data for fitting
    X_raw = []
    y_raw = []
    for df in datasets1:
        X_raw.append(df[input_feats].values)
        y_raw.append(df[output_feats].values)

    # 3. Pad sequences
    max_sequence_length = 5
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_raw, maxlen=max_sequence_length, padding='pre', truncating='pre', dtype='float32')
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(y_raw, maxlen=max_sequence_length, padding='pre', truncating='pre', dtype='float32')

    # X_raw is a list of arrays (timesteps x features) per sample
    flat_X = np.concatenate(X_raw, axis=0)  # Shape: [total_timesteps, n_features]
    X_min = flat_X.min(axis=0)              # Shape: [n_features]
    X_max = flat_X.max(axis=0)

    # 5. Build the model
    input_layer = tf.keras.Input(shape=(max_sequence_length, len(input_feats)), name="game_input")
    normalized_input = tf.keras.layers.LayerNormalization(name="layer_norm")(input_layer)
    lstm = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(normalized_input)
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(output_feats), activation='sigmoid'), name="rule_weights")(lstm)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    # 6. Train the model
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.33, shuffle=False)
    model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test))

    # 7. Save the model for serving
    model.save("app/lstm/rule_adjustment_model_0_0_4.keras", overwrite=False)

    return input_feats, output_feats, tf


@app.cell
def _(datasets1, input_feats, output_feats):
    import keras
    BATCH_SIZE = 3  # You can adjust the batch size
    dataset_timeseries = []
    for df_i in datasets1:
        dataset_timeseries.append(keras.utils.timeseries_dataset_from_array(df_i[input_feats], df_i[output_feats], BATCH_SIZE))
    # for dataset_t in dataset_timeseries:
    #     for batch in dataset_t:
    #         print(batch)
    print(dataset_timeseries)
    return (dataset_timeseries,)


@app.cell
def _(full_dataset):
    for x_batch, y_batch in full_dataset.take(9):
        print("X batch shape:", x_batch.shape)
        print("Y batch shape:", y_batch.shape)
    return


@app.cell
def _(dataset_timeseries, input_feats, output_feats, tf):
    full_dataset = dataset_timeseries[0]
    for ds in dataset_timeseries[1:]:
        full_dataset = full_dataset.concatenate(ds)

    # Define input and output shapes
    input_shape = (None, len(input_feats))
    output_shape = (len(output_feats),)  # Corrected output shape

    model1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input_from_metrics'),
        tf.keras.layers.LayerNormalization(name="layer_norm"),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=False, name='lstm'),
        tf.keras.layers.Dense(output_shape[0], activation='sigmoid') # Corrected Dense layer
    ])

    model1.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    # Batch and prefetch the dataset
    SHUFFLE_BUFFER_SIZE = 5

    batched_dataset = full_dataset.shuffle(SHUFFLE_BUFFER_SIZE).prefetch(tf.data.AUTOTUNE)

    model1.fit(batched_dataset, epochs=25)

    model1.save("app/lstm/rule_adjustment_model_0_0_5.keras", overwrite=False)
    return (full_dataset,)


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="save")
    save_location = mo.ui.text()
    mo.md(f'''
    Save dataset in: {save_location} \n
    Save preprocessed dataset? {save_button} \n
    ''')
    return save_button, save_location


@app.cell
def _(dataset_timeseries, mo, save_button, save_location):
    mo.stop(not save_button.value)

    for ds1_i, ds1 in enumerate(dataset_timeseries):
        ds1.save(f'{save_location.value}/dataset{ds1_i}')

    mo.md("Dataset saved.")
    return


if __name__ == "__main__":
    app.run()
