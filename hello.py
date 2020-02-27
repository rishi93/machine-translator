from flask import Flask
from flask import request
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

def add_start_stop_tokens(sentence):
    return '<start> ' + sentence + ' <end>'

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        output, state = self.gru(x)
        
        output = tf.reshape(output, (-1, output.shape[2]))
        
        x = self.fc(output)
        
        return x, state, attention_weights

def translate(sentence):
    sentence = add_start_stop_tokens(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, 1024))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tar_lang.word_index['<start>']], 0)
    
    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        result += tar_lang.index_word[predicted_id] + ' '
        
        if tar_lang.index_word[predicted_id] == '<end>':
            return result
        
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return result

with open(os.path.join(os.getcwd(), 'ML_files/inp_lang.pickle'), 'rb') as f:
    inp_lang = pickle.load(f)
with open(os.path.join(os.getcwd(), 'ML_files/tar_lang.pickle'), 'rb') as f:
    tar_lang = pickle.load(f)
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(tar_lang.word_index) + 1

max_length_inp = 8
max_length_tar = 12

encoder = Encoder(vocab_inp_size, 256, 1024, 64)
decoder = Decoder(vocab_tar_size, 256, 1024, 64)


encoder.load_weights(os.path.join(os.getcwd(), 'ML_files/encoder_weights'))
decoder.load_weights(os.path.join(os.getcwd(), 'ML_files/decoder_weights'))

@app.route('/translate')
def index():
    en_sentence = request.args.get('sentence')
    de_sentence = translate(en_sentence)
    return "German sentence: {}".format(de_sentence)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')