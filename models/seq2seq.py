import tensorflow as tf 
from models.encoder import Encoder
from models.decoder import Decoder

class Seq2Seq(tf.keras.Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, embedding_dim, units, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(enc_vocab_size, embedding_dim, units, num_layers)
        self.decoder = Decoder(dec_vocab_size, embedding_dim, units, num_layers)

    def call(self, enc_input, dec_input):
        enc_output , enc_states = self.encoder(enc_input)
        dec_output, _ = self.decoder(dec_input, enc_states)
        return dec_output