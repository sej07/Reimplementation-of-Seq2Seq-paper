import tensorflow as tf 
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, num_layers):
        super(Encoder, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [
            layers.LSTM(
                enc_units,
                return_sequences = True, 
                return_state = True,
                recurrent_initializer = 'glorot_uniform'
            ) for _ in range(num_layers)
        ]

    def call(self, x, hidden_states= None):
        x= self.embedding(x)
        states = []
        for lstm in self.lstm_layers:
            if hidden_states:
                x,h,c = lstm(x, initial_state = hidden_states)
            else:
                x, h,c = lstm(x)
            states.append((h,c))
        return x, states