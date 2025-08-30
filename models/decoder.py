import tensorflow as tf 
from tensorflow.keras import layers

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, num_layers):
        super(Decoder, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [
            layers.LSTM(
                dec_units,
                return_sequences = True,
                return_state = True,
                recurrent_initializer = 'glorot_uniform'
            ) for _ in range(num_layers)
        ]
        self.fc = layers.Dense(vocab_size)

    def call(self, x, hidden_states):
        x= self.embedding(x)
        for i, lstm in enumerate(self.lstm_layers):
            h,c = hidden_states[i]
            x, h,c = lstm(x, initial_state= [h,c])
            hidden_states[i] = (h,c)
        output = self.fc(x)
        return output, hidden_states