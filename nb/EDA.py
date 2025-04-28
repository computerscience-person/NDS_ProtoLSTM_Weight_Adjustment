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
    import pandas as pd
    import re

    def parse_log_content_flattened(content):
        """
        Parses the log content, extracts Rules Used ruleIDs and Script Generated weights,
        and structures data for a flattened table.
        """
        grouped_data = []
        all_generated_rule_ids = set()

        # Regex to find blocks of 'Script Generated' followed by 'Rules Used Log' with timestamp
        pattern = r'--- Script Generated ---\s*(\[.*?\])\s*--- Rules Used Log \| Timestamp: (.*?) ---\s*(\[.*?\])'
        matches = re.findall(pattern, content, re.DOTALL)

        for script_generated_json_str, timestamp_str, rules_used_json_str in matches:
            try:
                script_generated_data = json.loads(script_generated_json_str)
                rules_used_data = json.loads(rules_used_json_str)

                # Extract list of ruleIDs from Rules Used
                rules_used_ids = [rule.get('ruleID') for rule in rules_used_data if rule.get('ruleID') is not None]

                # Extract ruleID and weight from Script Generated and collect all unique rule IDs
                script_generated_weights = {}
                for rule in script_generated_data:
                    rule_id = rule.get('ruleID')
                    weight = rule.get('weight')
                    if rule_id is not None and weight is not None:
                        script_generated_weights[rule_id] = weight
                        all_generated_rule_ids.add(rule_id)

                grouped_entry = {
                    'Timestamp': timestamp_str,
                    'Rules Used': rules_used_ids,
                    'Script_Generated_Weights': script_generated_weights # Temporarily store weights in a nested dict
                }
                grouped_data.append(grouped_entry)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for timestamp {timestamp_str}: {e}")
            except Exception as e:
                print(f"An error occurred while processing timestamp {timestamp_str}: {e}")

        # Now, flatten the Script Generated weights into separate columns
        flattened_data = []
        sorted_generated_rule_ids = sorted(list(all_generated_rule_ids)) # Sort for consistent column order

        for entry in grouped_data:
            flattened_entry = {
                'Timestamp': entry['Timestamp'],
                'Rules Used': entry['Rules Used']
            }
            weights = entry['Script_Generated_Weights']
            for rule_id in sorted_generated_rule_ids:
                # Use get with a default of None in case a rule ID is not present in a specific timestamp's generated script
                flattened_entry[f'Weight_RuleID_{rule_id}'] = weights.get(rule_id, None)
            flattened_data.append(flattened_entry)

        return flattened_data

    # Specify the path to your uploaded file
    file_path = 'datasets/training.txt' # Make sure this matches the name of your uploaded file

    # Read the content of the file
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        exit()

    # Parse and flatten the data
    flattened_entries = parse_log_content_flattened(content)

    # Create a Pandas DataFrame
    df_flattened = pd.DataFrame(flattened_entries)

    # Set Timestamp as the index
    df_flattened.set_index('Timestamp', inplace=True)

    # Display the first few rows of the DataFrame
    print(df_flattened.head())

    # Save the DataFrame to a CSV file
    df_flattened.to_csv('formatted_training_data_flattened.csv')

    print("\nFormatted and flattened data saved to 'formatted_training_data_flattened.csv'")
    return (
        content,
        df_flattened,
        f,
        file_path,
        flattened_entries,
        json,
        parse_log_content_flattened,
        pd,
        re,
    )


@app.cell
def _(df_flattened):
    df_flattened
    return


@app.cell
def _(df_flattened, pd):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.pipeline import make_pipeline

    mlb = MultiLabelBinarizer()
    data_enc = mlb.fit_transform(df_flattened['Rules Used'])
    df_enc = pd.DataFrame(data_enc, columns=mlb.classes_)
    df = pd.concat([df_flattened.reset_index(), df_enc], axis=1)
    for n in [3, 4, 5, 13, 14, 16, 20]:
        df[n] = 0
    df.columns = [str(col) for col in df.columns]
    df = df[sorted(df.columns)]
    df
    return MultiLabelBinarizer, data_enc, df, df_enc, make_pipeline, mlb, n


@app.cell
def _(df):
    import numpy as np

    y = df.filter(regex='Weight_RuleID_[0-9]+')
    X = df.filter(regex='^[0-9]+$')
    y_data = df.filter(regex='Weight_RuleID_[0-9]+').values
    X_data = df.filter(regex='^[0-9]+$').values

    def create_sequences(input_data, output_data, sequence_length):
        sequences_X = []
        sequences_y = []
        for i in range(len(input_data) - sequence_length):
            seq_X = input_data[i : i + sequence_length]
            seq_y = output_data[i + sequence_length]  # Predict the output at the next time step
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        return np.array(sequences_X), np.array(sequences_y)

    sequence_length = 3  # Example sequence length
    X_sequences, y_sequences = create_sequences(X_data, y_data, sequence_length)

    X_sequences.shape, y_sequences.shape
    return (
        X,
        X_data,
        X_sequences,
        create_sequences,
        np,
        sequence_length,
        y,
        y_data,
        y_sequences,
    )


@app.cell
def _(X_sequences, sequence_length, y_sequences):
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.33, shuffle=False)

    input_layer = Input(shape=(sequence_length, X_sequences.shape[2]))
    lstm_layer = LSTM(64, activation='relu')(input_layer)
    output_layer = Dense(y_sequences.shape[1], activation='sigmoid')(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    return (
        Dense,
        Input,
        LSTM,
        Model,
        X_test,
        X_train,
        history,
        input_layer,
        keras,
        loss,
        lstm_layer,
        model,
        output_layer,
        train_test_split,
        y_test,
        y_train,
    )


@app.cell
def _(X_data, X_sequences, y_data, y_sequences):
    X_data.shape, y_data.shape, X_sequences.shape, y_sequences.shape
    return


@app.cell
def _(X, y):
    X.head(5), y.head(5)
    return


@app.cell
def _(model, np):
    # Test model prediction
    test_pred = np.array([[
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,],
    ]])

    model.predict(test_pred)
    return (test_pred,)


@app.cell
def _(model):
    model_filepath = "app/lstm/nds_0_0_1.keras"

    model.save(model_filepath)
    return (model_filepath,)


if __name__ == "__main__":
    app.run()
