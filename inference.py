import tensorflow as tf 
import pickle
import numpy as np
from models.seq2seq import Seq2Seq

embedding_dim = 1000
units= 1000
num_layers = 4
max_len_en= 2000
max_len_fr= 3000

with open("data/processed/en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load()
with open("data/processed/fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load()

enc_vocab_size = len(en_tokenizer.word_index) + 1
dec_vocab_size = len(fr_tokenizer.word_index) + 1

model = Seq2Seq(enc_vocab_size, dec_vocab_size, embedding_dim, units, num_layers)
model.build(input_shape=[(None, max_len_en), (None, max_len_fr)])

model.load_weights("checkpoints/final_model.h5")
print("Model weights loaded")

#preprocess english
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    seq = en_tokenizer.texts_to_sequences([sentence])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len_en, padding='post')
    return padded

def translate(sentence):
    input_seq = preprocess_sentence(sentence)
    # Encoder forward pass
    enc_output, enc_states = model.encoder(input_seq)
    # Decoder initial input: <sos>
    start_token = fr_tokenizer.word_index.get('<sos>', 1)
    end_token = fr_tokenizer.word_index.get('<eos>', 2)
    dec_input = tf.expand_dims([start_token], 0)
    states = enc_states
    result = []
    for _ in range(max_len_fr):
        preds, states = model.decoder(dec_input, states)
        pred_id = tf.argmax(preds[0, -1]).numpy()

        if pred_id == end_token:
            break
        word = fr_tokenizer.index_word.get(pred_id, '<unk>')
        result.append(word)
        dec_input = tf.expand_dims([pred_id], 0)
    return ' '.join(result)

if __name__ == "__main__":
    print("Enter a sentence in English (or 'exit' to quit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == 'exit':
            break
        output = translate(sentence)
        print(f"🇫🇷 Translation: {output}\n")