import os
import sys
sys.path.append("./")

# import librosa
# import librosa.filters
# import scipy
from params_dict import hparams
import numpy as np
from text import text_to_sequence
import cmudict
import random
# from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Flatten
# from text import symbols
# from keras.layers import Embedding
# from keras.models import Model
from util import audio
import io

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
#     for offset in tqdm(range(len(metadata))):
    meta = metadata[offset]
    text = meta[3]
    #if _cmudict and random.random() < p_cmudict:
    #    text = ' '.join([maybe_get_arpabet(word) for word in text.split(' ')])

    # print(text)
    spec = np.load(os.path.join(datadir, meta[0]))
    spec = spec.tolist()
    linear_targets.append(spec)
    input_texts.append(text_to_sequence(text, cleaner_names))

    return input_texts, linear_targets

def synthesize(linear_outputs):
    wav = audio.inv_spectrogram(linear_outputs)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()



if __name__ == '__main__':
    input_num = 101
    max_input_len = 0
    max_target_len = 0
    input_all = []
    targets_all = []
    input_text, linear_targets = get_examples(0)
    
    for i in range(input_num):
        input_texts, linear_targets = get_examples(i)
        if len(input_texts[0]) > max_input_len:
            max_input_len = len(input_texts[0])
        if len(linear_targets[0]) > max_target_len:
            max_target_len = len(linear_targets[0])
        input_all.append(input_texts[0])
        targets_all.append(linear_targets[0])
    
    for i in range(input_num):
        zero_num = max_input_len - len(input_all[i])
        for j in range(zero_num):
            input_all[i].append(0)
    
    # zero_list = []
    # for i in range(1025):
    #     zero_list.append(0)
    zero_list = [0 for i in range(1025)]

    for i in range(input_num):
        zero_num = max_target_len - len(targets_all[i])
        for j in range(zero_num):
            targets_all[i].append(zero_list)

    targets_all_array = np.asarray(targets_all)
    input_all_array = np.asarray(input_all)
    targets_array = np.zeros((input_num, max_target_len*1025))
    for i in range(input_num):
        targets_array[i,:] = np.reshape(targets_all_array[i], (1, max_target_len*1025))

    model = Sequential()
    model.add(Dense(max_input_len, input_dim= max_input_len, activation='relu'))
    model.add(Dense(max_input_len, activation='relu'))
    model.add(Dense(max_target_len*1025, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit(input_all_array[0:99, :], targets_array[0:99, :], epochs=100, batch_size=10)
    
    # index, slice, l[1:2], list[1]
    prediction = model.predict(input_all_array[100:, :])
    output = np.reshape(prediction, (max_target_len, 1025))
    '''wave = synthesize(output.T)
    with open('./test_wave.wav', "wb") as handle:
        handle.write(wave)'''

    
