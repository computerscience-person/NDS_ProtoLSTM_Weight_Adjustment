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
        Parses the log content, extracts Parameters, Script Generated weights,
        and Rules Used, and structures data for a flattened table.
        """
        grouped_data = []
        all_generated_rule_ids = set()

        # Regex to find blocks of 'Parameters', 'Script Generated', and 'Rules Used Log' with timestamp
        pattern = r'--- Parameters: (\{.*?\}\s*)---\s*--- Script Generated ---\s*(\[.*?\])\s*--- Rules Used Log(?: \| Timestamp: (.*?))? ---\s*(\[.*?\])'
        matches = re.findall(pattern, content, re.DOTALL)
        # print(matches)

        for params_json_str, script_generated_json_str, timestamp_str, rules_used_json_str in matches:
            try:
                params_data = json.loads(params_json_str)
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
                    'Parameters': params_data,
                    'Rules Used': rules_used_ids,
                    'Script_Generated_Weights': script_generated_weights  # Temporarily store weights in a nested dict
                }
                # grouped_entry.update(params_data)
                grouped_data.append(grouped_entry)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for timestamp {timestamp_str}: {e}")
            except Exception as e:
                print(f"An error occurred while processing timestamp {timestamp_str}: {e}")

        # Now, flatten the Script Generated weights into separate columns
        flattened_data = []
        sorted_generated_rule_ids = sorted(list(all_generated_rule_ids))  # Sort for consistent column order

        for entry in grouped_data:
            # print(entry)
            flattened_entry = {
                'Timestamp': entry['Timestamp'],
                'Rules Used': entry['Rules Used']
            }
            flattened_entry.update(flatten_parameters(entry))
            weights = entry['Script_Generated_Weights']
            for rule_id in sorted_generated_rule_ids:
                # Use get with a default of None in case a rule ID is not present in a specific timestamp's generated script
                flattened_entry[f'Weight_RuleID_{rule_id}'] = weights.get(rule_id, None)
            flattened_data.append(flattened_entry)

        return flattened_data

    def flatten_parameters(entry):
        """
        Flattens the 'Parameters' dictionary into the main entry.
        """
        flattened_entry = entry.copy()  # Create a copy to avoid modifying the original

        if 'Parameters' in flattened_entry and isinstance(flattened_entry['Parameters'], dict):
            parameters = flattened_entry.pop('Parameters')  # Remove 'Parameters' and get its value
            for key, value in parameters.items():
                flattened_entry[f'Parameter_{key}'] = value  # Add each parameter with a prefix

        return flattened_entry


    return flatten_parameters, json, parse_log_content_flattened, pd, re


@app.cell
def _(parse_log_content_flattened, pd):
    def format_save(file_path: str) -> None :
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

        save_path = f'{file_path.replace('.', '_')}_formatted.csv'
        df_flattened.to_csv(save_path)

        print(f"\nFormatted and flattened data saved to '{save_path}'")

    dfs = []

    for n in range(1, 10):
        format_save(f'datasets/set2/training{n}.txt')
        dfs.append(pd.read_csv(f'datasets/set2/training{n}_txt_formatted.csv'))
    return dfs, format_save, n


@app.cell
def _(pd):
    # 23 cols, fewer weights
    df1 = pd.read_csv('datasets/set2/training1_txt_formatted.csv')
    df2 = pd.read_csv('datasets/set2/training2_txt_formatted.csv')
    df3 = pd.read_csv('datasets/set2/training3_txt_formatted.csv')
    df4 = pd.read_csv('datasets/set2/training4_txt_formatted.csv')
    df5 = pd.read_csv('datasets/set2/training5_txt_formatted.csv')
    # 31 cols, more weights
    df6 = pd.read_csv('datasets/set2/training6_txt_formatted.csv')
    df7 = pd.read_csv('datasets/set2/training7_txt_formatted.csv')
    df8 = pd.read_csv('datasets/set2/training8_txt_formatted.csv')
    df9 = pd.read_csv('datasets/set2/training9_txt_formatted.csv')

    # dfs # 6-9 only relevant
    return df1, df2, df3, df4, df5, df6, df7, df8, df9


@app.cell
def _(dfs, pd, re):
    import numpy as np
    from ast import literal_eval
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.pipeline import make_pipeline

    def df_prep1(df_flattened: pd.DataFrame) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()
        rules = df_flattened['Rules Used'].apply(literal_eval)
        data_enc = mlb.fit_transform(rules)
        df_enc = pd.DataFrame(data_enc, columns=mlb.classes_)
        df = pd.concat([df_flattened.reset_index(), df_enc], axis=1)
        df.columns = [str(col) for col in df.columns]
        df = df[sorted(df.columns)]
        return df

    dfs_prepped = []

    for dfn in dfs:
        dfs_prepped.append(df_prep1(dfn))

    pat_ruleids = r'Weight_RuleID_([0-9]+)'
    for nx in range(5, 9):
        new_cols = []
        for col in dfs_prepped[nx]:
            matches_ruleids = re.match(pat_ruleids, col)
            if matches_ruleids: 
                for match in matches_ruleids.groups():
                    if match not in dfs_prepped[nx].columns:
                        # new_cols.append(match)
                        dfs_prepped[nx][f'{match}'] = 0
        dfs_prepped[nx].columns = [str(col) for col in dfs_prepped[nx].columns]
        dfs_prepped[nx] = dfs_prepped[nx][sorted(dfs_prepped[nx].columns)]

    # df6.columns
    # dfs_prepped[5:9]
    # set(dfs_prepped[7].columns) == set(dfs_prepped[8].columns)
    # dfs_prepped[7]
    return (
        MultiLabelBinarizer,
        col,
        df_prep1,
        dfn,
        dfs_prepped,
        literal_eval,
        make_pipeline,
        match,
        matches_ruleids,
        new_cols,
        np,
        nx,
        pat_ruleids,
    )


@app.cell
def _(dfs_prepped, np):
    def split_dsets(df):
        y = df.filter(regex='Weight_RuleID_[0-9]+')
        X = df.filter(regex=r'^[0-9]+$|Parameter\w+')
        y_data = df.filter(regex='Weight_RuleID_[0-9]+').values
        X_data = df.filter(regex=r'^[0-9]+$|Parameter\w+').values
        return X, X_data, y, y_data

    dfs_split = [split_dsets(df.ffill().bfill()) for df in dfs_prepped[5:9]]

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
    input_seq = [create_sequences(df_split[1], df_split[3], sequence_length) for df_split in dfs_split]
    # print(*(nm[0].shape for nm in input_seq))
    Xs = [item[0] for item in input_seq]
    X = np.concatenate([arr for arr in Xs])
    ys = [item[1] for item in input_seq]
    y = np.concatenate([arr for arr in ys])
    X.shape, y.shape
    return (
        X,
        Xs,
        create_sequences,
        dfs_split,
        input_seq,
        sequence_length,
        split_dsets,
        y,
        ys,
    )


@app.cell
def _(X, sequence_length, y):
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    input_layer = Input(shape=(sequence_length, X.shape[2]))
    lstm_layer = LSTM(64, activation='relu')(input_layer)
    output_layer = Dense(y.shape[1], activation='sigmoid')(lstm_layer)
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
def _(dfs_split):
    dfs_split[0][0].columns

    # test_pred = np.array([[
    #     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #     [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,],
    # ]])

    # model.predict(test_pred)
    return


if __name__ == "__main__":
    app.run()
