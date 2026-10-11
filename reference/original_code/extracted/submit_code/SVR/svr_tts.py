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
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import io
import tensorflow as tf
from librosa import effects
from util import audio



def maybe_get_arpabet(word):
    _cmudict = cmudict.CMUDict('./cmudict-0.7b', keep_ambiguous=False)
    arpabet = _cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word

def get_examples(offset,metadata_filename='./training/training.txt'):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    offset = 0
    cleaner_names = [x.strip() for x in hparams['cleaners'].split(',')]
    p_cmudict = 0.5
    _cmudict = cmudict.CMUDict('./cmudict-0.7b', keep_ambiguous=False)
    datadir = os.path.dirname(metadata_filename)

    with open(metadata_filename, encoding='utf-8') as f:
      metadata = [line.strip().split('|') for line in f]
    random.shuffle(metadata)

    input_texts = []
    linear_targets = []

    for offset in tqdm(range(len(metadata))):
      meta = metadata[offset]
      text = meta[3]
      if _cmudict and random.random() < p_cmudict:
       text = ' '.join([maybe_get_arpabet(word) for word in text.split(' ')])

      spec = np.load(os.path.join(datadir, meta[0]))
      spec = spec.tolist()
      linear_targets.append(spec)
      input_texts.append(text_to_sequence(text, cleaner_names))

    return input_texts, linear_targets


def get_examples_data(text):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    cleaner_names = [x.strip() for x in hparams['cleaners'].split(',')]
    p_cmudict = 0.5
    _cmudict = cmudict.CMUDict('./cmudict-0.7b', keep_ambiguous=False)
    input_texts = []
    input_texts.append(text_to_sequence(text, cleaner_names))

    return input_texts

def read_data(input_texts,linear_targets):
    max_i = 0
    max_tar = 0
    for i in range(0,len(input_texts)):
        if(max_i < len(input_texts[i])):
            max_i = len(input_texts[i])
        if(max_tar < len(linear_targets[i])):
            max_tar = len(linear_targets[i])

    for i in range(0,len(input_texts)):
        z1 = np.zeros((1,max_i - len(input_texts[i])))
        input_texts[i] = np.concatenate((input_texts[i],z1), axis=None)
        z2 = np.zeros((max_tar - len(linear_targets[i]),1025))
        linear_targets[i] = np.concatenate((linear_targets[i],z2), axis=0)

    return input_texts,linear_targets

def read_data_t(input_texts,linear_targets,max_i,max_tar):
    for i in range(0,len(input_texts)):
        z1 = np.zeros((1,max_i - len(input_texts[i])))
        input_texts[i] = np.concatenate((input_texts[i], z1), axis=None)
        z2 = np.zeros((max_tar - len(linear_targets[i]),1025))
        linear_targets[i] = np.concatenate((linear_targets[i],z2), axis=0)

    return input_texts,linear_targets

def read_data_s(input_texts,max_i):
    for i in range(0,len(input_texts)):
        z1 = np.zeros((1,max_i - len(input_texts[i])))
        input_texts[i] = np.concatenate((input_texts[i], z1), axis=None)

    return input_texts

def fit_svm_model(model, input_texts,linear_targets):
    for m in range(0,len(linear_targets[0])):
        svc_model = SVR(kernel = 'poly',gamma='scale', C=1.0, epsilon=0.2)
        #svc_model = linear_model.PassiveAggressiveRegressor(max_iter = 1000,tol=1e-03)
        svc_m = MultiOutputRegressor(svc_model)
        x_t = []
        y_t = []
        for i in range(0,len(input_texts)):
            x_t.append(input_texts[i])
            y_t.append(linear_targets[i][m])
        x_train = np.array(x_t)
        y_train = np.array(y_t)
        svc_m.fit(x_train,y_train)
        model.append(svc_m)
    return model

def predict_svm_model(model,input_texts,n):
    linear_targets = np.zeros((1,1025))
    for i in range(0,n):
        a = model[i].predict(input_texts)
        linear_targets = np.concatenate((linear_targets,a), axis=0)

    return np.array(linear_targets)


def synthesize(linear_outputs):
    wav = audio.inv_spectrogram(linear_outputs)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()

if __name__ == '__main__':
    input_texts, linear_targets = get_examples(0)
    input_texts_n,linear_targets_n = read_data(input_texts,linear_targets)
    model = []
    n = len(linear_targets_n[0])
    m = len(input_texts_n[0])
    model1 = fit_svm_model(model,input_texts_n,linear_targets_n)
    #test_text, target = get_examples(0,'./training/test_file.txt')
    #test_text, target = read_data_t(test_text,target,m,n)
    test_text  = get_examples_data('The same evils of overcrowding, uncleanliness, want of medical attendance')
    test_text = read_data_s(test_text,m)
    target_pred = predict_svm_model(model1,test_text,n)
    #print("(", len(target_pred[1]), len(target_pred[1][0]), ")")
    print(target_pred.shape)
    wave = synthesize(target_pred.T)
    with open('./test_wave.wav', "wb") as handle:
            handle.write(wave)
