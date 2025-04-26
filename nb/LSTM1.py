import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import keras
    from keras import layers, ops

    inputs = keras.Input(shape=(12,2))
    # dense = layers.Dense(32)
    lstm = layers.LSTM(12)
    softmax = layers.Softmax()

    x1 = lstm(inputs)
    # x2 = lstm(x1)
    outputs = softmax(x1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="single_layer_lstm")
    model.summary()
    return inputs, keras, layers, lstm, model, ops, outputs, softmax, x1


@app.cell
def _(keras, model):
    keras.utils.plot_model(model, "lstm_2.png", show_shapes=True)
    return


if __name__ == "__main__":
    app.run()
