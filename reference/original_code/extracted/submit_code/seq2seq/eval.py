import sys
import os
import io
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
from util import audio

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import load_model
from train import generate_np, get_examples

def decode_sequence(input_seq,start_p, max_decoder_seq_length, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # print(states_value)

    # Generate empty target sequence of length 1.
    # target_seq = np.zeros((1, 1, 1025))
    target_seq = start_p.reshape((1,-1,1025))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_wave = np.zeros((1,1,1025))
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)


        # Sample a token
        decoded_wave = np.append(decoded_wave, output_tokens, axis=1)

        # Exit condition: either hit max length
        # or find stop character.
        if decoded_wave.shape[1] > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        # target_seq = np.zeros((1, 1, 1025))
        # target_seq[0, 0, :]= output_tokens
        target_seq = output_tokens

        # Update states
        states_value = [h, c]

    return decoded_wave

def systhesis(wav):
    wav = audio.inv_spectrogram(wav.T)
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()


import seq2seq
from seq2seq.models import AttentionSeq2Seq
from keras.callbacks import TensorBoard
from time import time

if __name__ == '__main__':
    # input_text, linear_target  = get_examples(300)

    encoder_input = np.load('encoder_input.npy')
    decoder_input = np.load('decoder_input.npy')

    # encoder_model = load_model('test_encoder_model.h5')
    # decoder_model = load_model('test_decoder_model.h5')
    max_encoder_seq_length = encoder_input.shape[1]
    max_decoder_seq_length = decoder_input.shape[1]

    # test_input = np.zeros((1, max_encoder_seq_length))
    # test_input[0, :len(input_text)] = input_text
    # wave_matrix = np.zeros((1, max_decoder_seq_length, 1025))
    # wave_matrix[0, :len(linear_target), :] = linear_target

    # print(wave_matrix.shape)
    # print(test_input.shape)

    '''
    # model part

    encoder_inputs = Input(shape=(max_encoder_seq_length, ))
    embed = Embedding(len(symbols), 1025, input_length=max_encoder_seq_length)

    embed_layer = embed(encoder_inputs)

    atten_model = AttentionSeq2Seq(input_dim=1025, input_length=max_encoder_seq_length, hidden_dim=256, output_length=max_decoder_seq_length, output_dim=1025, depth=4)

    final_output = atten_model(embed_layer)

    model = Model(encoder_inputs, final_output)

    # model part
    '''

    text_input = encoder_input[199:200, :]
    decoder_input = decoder_input[197:198, :, :]
    # model.load_weights('attention_lstm.h5')
    model = load_model('test.h5')
    decoded_wave = model.predict([text_input, decoder_input])
    print(text_input.shape)
    print(decoded_wave.shape)

    with open('./test_wave.wav', 'wb') as f:
        f.write(systhesis(decoded_wave[0]))
    '''
    from train import create_model

    model, encoder_model, decoder_model = create_model(max_encoder_seq_length, max_decoder_seq_length, 1025, symbols)

    # test1 = encoder_model.predict(encoder_input[0:1,:])
    # test3 = encoder_model.predict(encoder_input[0:1,:])
    encoder_model.load_weights('test.h5', by_name=True)
    decoder_model.load_weights('test.h5', by_name=True)
    # test2 = encoder_model.predict(encoder_input[0:1, :])
    # print(test1[0] == test2[0])
    # print(test3[0] == test1[0])

    model = load_model('test.h5')
    # model.summary()
    print("###############################")
    # encoder_model.summary()
    # print(encoder_model.get_weights())
    print("###############################")
    # decoder_model.summary()


    for seq_index in range(1):
	# Take one sequence (part of the training set)
	# for trying out decoding.
        input_seq = encoder_input[seq_index: seq_index + 1, :]
        # print(input_seq.shape)
        start_p = decoder_input[seq_index:seq_index+1, 0:100, :]
        decoded_wave = decode_sequence(input_seq, start_p, max_decoder_seq_length, encoder_model, decoder_model)
        print('-')
	# print('Input sentence:', input_texts[seq_index])
        print('Decoded wave:', decoded_wave.shape)
        print(decoded_wave)

    # decoded_wave = model.predict([encoder_input[0:1,:], decoder_input[0:1,:,:]])
    # print(decoded_wave.shape)
    with open('test.wav', 'wb') as f:
       f.write(systhesis(decoded_wave[0]))
    '''
