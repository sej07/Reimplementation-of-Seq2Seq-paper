import numpy as np
import pickle
import tensorflow as tf 
from models.seq2seq import Seq2Seq
import math

#data loading
en_input = np.load("data/processed/en_sequence.npy")
fr_output = np.load("data/processed/fr_sequence.npy")

with open("data/processed/en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)
with open("data/processed/fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load(f)

#tf dataset with batching
BATCH_SIZE = 128
BUFFER_SIZE = len(en_input)
dataset = tf.data.Dataset.from_tensor_slices((en_input, fr_output))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder  = True)

#model creation 
enc_vocab_size = len(en_tokenizer.word_index) + 1
dec_vocab_size = len(fr_tokenizer.word_index) + 1
embedding_dim = 1000
units = 1000
num_layers = 4
model = Seq2Seq(enc_vocab_size, dec_vocab_size, embedding_dim, units, num_layers)

#loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.7)

#training loop with teacher forcing
def train_step(enc_input, dec_target):
    loss= 0
    dec_input = dec_target[:, :-1] #remove <eos>
    real = dec_target[:, 1:] #remove <sos>

    with tf.GradientTape() as tape:
        predictions = model(enc_input, dec_input)
        loss = loss_object(real, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

EPOCHS = 7
steps_per_epoch = len(en_input) // BATCH_SIZE
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    if epoch > 5:
        lr = 0.7 * (0.5 ** (2 * (epoch -5)))
        optimizer.learning_rate.assign(lr)

    print(f"\nEpoch {epoch} / {EPOCHS} - LR: {optimizer.learning_rate.numpy():.4f}")
    for step, (enc_batch , dec_batch) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(enc_batch, dec_batch)
        total_loss += batch_loss

        if step % 100 == 0:
            print(f" Step {step}/{steps_per_epoch} - Batch loss: {batch_loss.numpy():.4f}")
    
    avg_loss = total_loss / steps_per_epoch
    model.save_weights(f"checkpoints/epoch_{epoch}.h5")
    print(f"Epoch {epoch} Loss: {avg_loss.numpy():.4f}")