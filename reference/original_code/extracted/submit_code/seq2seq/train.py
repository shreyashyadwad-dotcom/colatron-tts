import sys
import os
sys.path.append("./")

import librosa
import librosa.filters
import scipy
from params_dict import hparams
import numpy as np
from text import text_to_sequence
import cmudict
import random
from tqdm import tqdm
from text import symbols

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, activations
from keras.layers.recurrent import Recurrent
from keras import backend as K
from keras.engine import InputSpec

from tensorflow.keras.layers import concatenate

def _time_distributed_dense(x, w, b=None, dropout=None,
                           input_dim=None, output_dim=None, timesteps=None):
    '''Apply y.w + b for every temporal slice y of x.
    '''
    if not input_dim:
        # won't work with TensorFlow
        input_dim = K.shape(x)[2]
    if not timesteps:
        # won't work with TensorFlow
        timesteps = K.shape(x)[1]
    if not output_dim:
        # won't work with TensorFlow
        output_dim = K.shape(w)[1]

    if dropout:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x *= expanded_dropout_matrix

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))

    x = K.dot(x, w)
    if b:
        x = x + b
    # reshape to 3D tensor
    x = K.reshape(x, (-1, timesteps, output_dim))
    return x

def maybe_get_arpabet(word):
    _cmudict = cmudict.CMUDict('./cmudict-0.7b', keep_ambiguous=False)
    arpabet = _cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word

def get_examples(offset, metadata_filename='./training/train.txt'):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    cleaner_names = [x.strip() for x in hparams['cleaners'].split(',')]
    p_cmudict = 0.5
    _cmudict = cmudict.CMUDict('./cmudict-0.7b', keep_ambiguous=False)
    datadir = os.path.dirname(metadata_filename)

    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
    random.shuffle(metadata)
    
    input_texts = []
    linear_targets = []
    meta = metadata[offset]
    text = meta[3]
    if _cmudict and random.random() < p_cmudict:
        text = ' '.join([maybe_get_arpabet(word) for word in text.split(' ')])

    spec = np.load(os.path.join(datadir, meta[0]))
    # print(spec.shape)
    spec = spec.tolist()
    linear_targets.append(spec)
    input_texts.append(text_to_sequence(text, cleaner_names))

    return text_to_sequence(text, cleaner_names), spec
	
def read_data(num):
    input_texts = []
    linear_target = []
    max_encoder_seq_length = 0
    max_decoder_seq_length = 0
    for i in tqdm(range(num)):
        text, spec = get_examples(i)
        input_texts.append(text)
        linear_target.append(spec)
        
        if len(input_texts[i]) > max_encoder_seq_length:
            max_encoder_seq_length = len(input_texts[i])
            
        if len(linear_target[i]) > max_decoder_seq_length:
            max_decoder_seq_length = len(linear_target[i])
        
    return input_texts, linear_target, max_encoder_seq_length, max_decoder_seq_length
	
def generate_np(input_texts, linear_targets, max_encoder_seq_length, max_decoder_seq_length):

    encoder_input = np.zeros((len(input_texts), max_encoder_seq_length), dtype='float32')
    decoder_input = np.zeros((len(input_texts), max_decoder_seq_length, len(linear_targets[0][0])), dtype='float32')
    decoder_target = np.zeros((len(input_texts), max_decoder_seq_length, len(linear_targets[0][0])), dtype='float32')

    for i, (text, target) in enumerate(zip(input_texts, linear_targets)):
        encoder_input[i, :len(input_texts[i])] = np.array(input_texts[i])
        decoder_input[i, :len(linear_targets[i]), :] = np.array(linear_targets[i])
        decoder_target[i, :-1, :] = decoder_input[i, 1:, :]
        
    return encoder_input, decoder_input, decoder_target

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

class AttentionDecoder(Recurrent):
    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space
 
        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
 
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
 
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
 
        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences
 
    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """
 
        self.batch_size, self.timesteps, self.input_dim = input_shape
 
        if self.stateful:
            super(AttentionDecoder, self).reset_states()
 
        # self.states = [None, None]  # y, s
 
        """
            Matrices for creating the context vector
        """
 
        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
 
        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
 
        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
 
        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True
 
    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x
 
        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)
 
        return super(AttentionDecoder, self).call(x)
 
    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))
 
        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])
 
        return [y0, s0]
 
    def step(self, x, states):
 
        ytm, stm = states
 
        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)
 
        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)
 
        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)
 
        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:
 
        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)
 
        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)
 
        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)
 
        # new hidden state:
        st = (1-zt)*stm + zt * s_tp
 
        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)
 
        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]
 
    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)
 
    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def create_model(max_encoder_seq_length, max_decoder_seq_length, audio_depth, symbols, latent_dim=512):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(max_encoder_seq_length, ))
    embed = Embedding(len(symbols), audio_depth, input_length=max_encoder_seq_length)
    
    embed_layer = embed(encoder_inputs)
    
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(embed_layer)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, audio_depth))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs_before, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    
    # 512 Dense
    h_dense = Dense(512, activation='relu')
    decoder_outputs_before = h_dense(decoder_outputs_before)

    decoder_dense = Dense(audio_depth, activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs_before)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
	
def create_attention_model(max_encoder_seq_length, max_decoder_seq_length, audio_depth, symbols, latent_dim=512):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(max_encoder_seq_length, ))
    embed = Embedding(len(symbols), audio_depth, input_length=max_encoder_seq_length)
    
    embed_layer = embed(encoder_inputs)
    
    encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, state_h, state_c = encoder(embed_layer)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # attention = AttentionDecoder(latent_dim, latent_dim)(encoder_outputs)
    attention = Attention(latent_dim)(encoder_outputs)
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, audio_depth))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    # decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    merge = concatenate([attention, decoder_input]) 
    
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs_before, _, _ = decoder_lstm(merge, # decoder_inputs,
                                         initial_state=encoder_states)
    
    # 512 Dense
    h_dense = Dense(512, activation='relu')
    decoder_outputs_before = h_dense(decoder_outputs_before)

    decoder_dense = Dense(audio_depth, activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs_before)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

import seq2seq
from seq2seq.models import AttentionSeq2Seq   
from keras.callbacks import TensorBoard 
from time import time

from util import audio
import io

def systhesis(wav):
    wav = audio.inv_spectrogram(wav.T)
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()

if __name__ == '__main__':
    '''
    input_texts, linear_target, max_encoder_seq_length, max_decoder_seq_length = read_data(1)
    encoder_input, decoder_input, decoder_target = generate_np(input_texts, linear_target, max_encoder_seq_length, max_decoder_seq_length)
    
    '''
    encoder_input = np.load('encoder_input.npy')
    decoder_input = np.load('decoder_input.npy')
    decoder_target = np.load('decoder_target.npy')
    max_encoder_seq_length = encoder_input.shape[1]
    max_decoder_seq_length = decoder_target.shape[1]
    
    temp1 = np.zeros((200, 200))
    temp2 = np.zeros((200, 1000, 1025))
    temp1[:, :max_encoder_seq_length] = encoder_input
    temp2[:, :max_decoder_seq_length, :] = decoder_target
    encoder_input = temp1
    decoder_target = temp2
    
    
    max_encoder_seq_length = 200
    max_decoder_seq_length = 1000
    
    audio_depth = 1025
    
    # model part
    
    encoder_inputs = Input(shape=(max_encoder_seq_length, ))
    embed = Embedding(len(symbols), 1025, input_length=max_encoder_seq_length)
    
    embed_layer = embed(encoder_inputs)
    
    # decoder_inputs = Input(shape=(None, audio_depth))
    
    atten_model = AttentionSeq2Seq(input_dim=1025, input_length=max_encoder_seq_length, hidden_dim=256, output_length=max_decoder_seq_length, output_dim=1025, depth=8)
    
    final_output = atten_model(embed_layer)
    
    model = Model(encoder_inputs, final_output)
    
    # model part
    
    # from keras.models import load_weights
    # model.load_weights('attention_lstm.h5')
    
    y = decoder_target
    print(encoder_input.shape)
    print(decoder_target.shape)
    
    # model, encoder_model, decoder_model = create_attention_model(max_encoder_seq_length, max_decoder_seq_length, 1025, symbols)
    
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    
    offset = 1
    text, spec = get_examples(offset)
    input_data = np.zeros((1, 200))
    input_data[0, 0:len(text)] = text
    
    output_data = np.zeros((1, 1000, 1025))
    output_data[0, 0:len(spec), :] = spec
    
    for i in range(100000):
      
      '''
      # offset = i % 2000
      text, spec = get_examples(offset)
      input_data = np.zeros((1, 200))
      input_data[0, 0:len(text)] = text
      
      output_data = np.zeros((1, 1000, 1025))
      output_data[0, 0:len(spec), :] = spec
      
      scores = model.evaluate(encoder_input[190:,:], decoder_target[190:, :], verbose=0)
      print("test: ", scores)
      '''
    
      if i % 50 == 0 and i != 0:
      
          # file_name = "model_epoch_" + str(i) + ".h5"
          # model.save('./large_models/'+file_name)
          # decode_train = model.predict(encoder_input[0:1, :])
          decode_train = model.predict(input_data)
          with open('./large_models/attention_train'+str(i)+'.wav', 'wb') as f:
              f.write(systhesis(decode_train[0]))
          '''
          decode_test = model.predict(encoder_input[195:196, :])
          with open('./large_models/attention_test'+str(i)+'.wav', 'wb') as f:
              f.write(systhesis(decode_test[0]))
          '''

      # for j in range(0, 180, 10):
        # model.fit(encoder_input[j:j+10, :], decoder_target[j:j+10, :, :], epochs=1, verbose=2)
      model.fit(input_data, output_data, epochs=1, verbose=2)
    
    # model.fit(encoder_input, decoder_target, epochs=1)# verbose=)
    
    '''
    np.save('encoder_input_1000.npy', encoder_input)
    np.save('decoder_input_1000.npy', decoder_input)
    np.save('decoder_target_1000.npy', decoder_target)
    
    # encoder_input = np.load('encoder_input.npy')
    # decoder_input = np.load('decoder_input.npy')
    # decoder_target = np.load('decoder_target.npy')

    max_encoder_seq_length = encoder_input.shape[1]
    max_decoder_seq_length = decoder_input.shape[1]

    model, encoder_model, decoder_model = create_model(max_encoder_seq_length, max_decoder_seq_length, 1025, symbols)
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    model.fit([encoder_input, decoder_input], decoder_target, epochs=250, verbose=2)
    model.save('test_1000.h5')
	
    encoder_model.load_weights('test_1000.h5', by_name=True)
    decoder_model.load_weights('test_1000.h5', by_name=True)
	
    encoder_model.save('test_encoder_model_1000.h5')
    decoder_model.save('test_decoder_model_1000.h5')
    '''
    
