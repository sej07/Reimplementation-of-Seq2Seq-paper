import unicodedata
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

#load english french pairs
def load_raw_sentence_pairs(file_path):
    pairs= []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            eng = parts[0].strip()
            fr = parts[1].strip()
            pairs.append((eng, fr))
    return pairs

#Normalize sentences
def normalize_sentence(sentence):
    sentence = unicodedata.normalize('NFD', sentence)
    sentence = sentence.encode('ascii', 'ignore').decode('utf-8')
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()

#reverse english and add <sos> and <eos>
def process_sentence_pairs(pairs):
    processed_en = []
    processed_fr= []
    for en,fr in pairs:
        en = normalize_sentence(en)
        fr= normalize_sentence(fr)
        en_tokens = en.split()[::-1]
        fr_tokens = fr.split()
        en_processed = ' '.join(en_tokens)
        fr_processed = '<sos> ' + ' '.join(fr_tokens) +' <eos>'
        processed_en.append(en_processed)
        processed_fr.append(fr_processed)
    return processed_en, processed_fr

#tokenize sentences
def build_tokenizer(sentences, num_words = None, filters=''):
    tokenizer = Tokenizer(num_words= num_words, filters= filters, oov_token = '<unk>')
    tokenizer.fit_on_texts(sentences)
    return tokenizer

#padding
def sentences_to_padded_sequences(tokenizer, sentences, maxlen = None):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding = 'post', maxlen = maxlen)
    return padded

#save tokenizer and processed data
def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def save_numpy_array(array, path):
    np.save(path, array)

if __name__ == "__main__":
    path = 'data/raw/fra.txt'
    raw_pairs= load_raw_sentence_pairs(path)
    en_list, fr_list = process_sentence_pairs(raw_pairs)
    en_tokenizer = build_tokenizer(en_list)
    fr_tokenizer = build_tokenizer(fr_list)
    en_seq = sentences_to_padded_sequences(en_tokenizer, en_list)
    fr_seq = sentences_to_padded_sequences(fr_tokenizer, fr_list)
    
    save_tokenizer(en_tokenizer, 'data/processed/en_tokenizer.pkl')
    save_tokenizer(fr_tokenizer, 'data/processed/fr_tokenizer.pkl')

    save_numpy_array(en_seq, 'data/processed/en_sequence.npy')
    save_numpy_array(fr_seq, 'data/processed/fr_sequence.npy')
    print("Tokenizer and sequences saved")