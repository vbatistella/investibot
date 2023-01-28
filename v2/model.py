from statistics import mode
from scipy.fftpack import shift
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras

class Model():
    # Creates LTSM with dense output
    def __init__(self, input, output, shift):
        self.input = input
        self.output = output
        self.shift = shift

        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(input, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(input, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(input, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(input, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output)
        ])
        self.model = model
        # compile with MSE as loss and ADAM as optimizer
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    
    # Train the model
    def train(self, x_train, y_train, epochs, batch):
        # Define callback for early stop
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
        # Do the actual training
        history = self.model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
        return history
    
    # Save
    def save(self, name):
        self.model.save(name)

    # Measure performance on testing set   
    def performance(self, x, y):
        return self.model.evaluate(x, y)
    
    # Infer from input
    def infer(self, input):
        return self.model.predict(input)
    
    # Load model from filename
    def load(self, name):
        self.model = keras.models.load_model(name)
