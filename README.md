## Seq2Seq Reimplementation (TensorFlow & Keras)
_This project reimplements the original Sequence-to-Sequence Learning with Neural Networks (Sutskever et al., 2014) paper using TensorFlow and Keras. The model demonstrates how multilayered LSTMs can learn meaningful phrase and sentence representations for English-to-French translation, with word-order sensitivity and effective decoding strategies._

#### Paper Highlights:
- Encoder-Decoder Architecture
    1. Multilayered LSTM (4 layers) in both encoder and decoder.
    2. Each LSTM layer contains 1000 hidden units.
    3. Word embeddings of size 1000.
- Model Highlights
    1. Source sentences reversed before feeding into the encoder.
    2. Decoder trained using Teacher Forcing.
    3. Inference decoding via Beam Search (width = 12).
    4. Special tokens used: <EOS> for end of sentence, <UNK> for unknown words.

#### Dataset Details:
- WMT’14 English-French parallel corpus.
- Input vocabulary: 160,000 words.
- Output vocabulary: 80,000 words.
- Naive softmax over 80k words at each output step.

#### Repository Structure
- data/ : Preprocessed and Raw dataset
- models/ : Encoder, Decoder, Seq2Seq implementation
- Seq2Seq/ : Packages and Scripts
- utils/ : Configuration and Preprocessing
- evaluate.py
- inference.py
- predict.py
- train.py

#### Training Setup: 
- Optimizer: Stochastic Gradient Descent (no momentum).
- Gradient clipping applied to stabilize training.
- Initial learning rate: 0.7.
- After 5 epochs → learning rate halved every half epoch.
- Total training: 7.5 epochs.
- Batch size: 128.

#### Performance:
- Achieved a BLEU score of 34.81 on English-to-French translation.

#### Frameworks:
- Tensorflow and Keras

#### Workflow: 
1. Tokenization and preprocessing of parallel sentences.
2. Reversing English input before feeding to encoder.
3. Encoder LSTM processes input one timestep at a time.
4. Decoder LSTM generates target output sequences with Teacher Forcing.
5. Inference uses Beam Search for decoding.

#### Improvements
1. Replace naive softmax with a more efficient sampled softmax for large vocabularies.
2. Experiment with Adam optimizer and compare against SGD.

#### Assumptions
1. The dataset preprocessing follows the original paper’s setup, where unknown words are mapped to <UNK>.
2. The vocabulary sizes (160k input, 80k output) are fixed and not dynamically pruned.
3. Training hyperparameters (learning rate schedule, SGD without momentum) were kept identical to the paper for fidelity, even though modern optimizers like Adam could improve stability.  

#### Key Observation
Reversing the source input sentences before encoding significantly improved translation quality, as it shortens the dependencies the LSTM must learn.

