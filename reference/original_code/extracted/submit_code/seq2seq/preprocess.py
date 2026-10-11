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


def maybe_get_arpabet(word):
    _cmudict = cmudict.CMUDict('./cmudict-0.7b', keep_ambiguous=False)
    arpabet = _cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word

def get_examples(offset, metadata_filename='./training/train.txt'):
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
#     for offset in tqdm(range(len(metadata))):
    meta = metadata[offset]
    text = meta[3]
    if _cmudict and random.random() < p_cmudict:
        text = ' '.join([maybe_get_arpabet(word) for word in text.split(' ')])

    # print(text)
    spec = np.load(os.path.join(datadir, meta[0]))
    spec = spec.tolist()
    linear_targets.append(spec)
    input_texts.append(text_to_sequence(text, cleaner_names))

    return input_texts, linear_targets

if __name__ == '__main__':
    for offset in range(3):
        input_texts, linear_targets = get_examples(offset)
        print(len(input_texts[0]))
        print(len(linear_targets[0]), len(linear_targets[0][0]))

    '''
    for i in range(len(input_texts)):
        print("input_texts   : ", "No. ", i, len(input_texts[i]))
        print("linear_targets: ", "No. ", i, "(", len(linear_targets[i]), len(linear_targets[i][0]), ")")
    '''
