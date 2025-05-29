#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.8": *

#set page(width: auto, height: auto, margin: 0.5em)

#codly(languages: codly-languages)
```py
tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape, name='input_from_metrics'),
    tf.keras.layers.LayerNormalization(name="layer_norm"),
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=False, name='lstm'),
    tf.keras.layers.Dense(output_shape[0], activation='sigmoid')
])
```
