from nltk.corpus import cmudict

import numpy as np
import pandas as pd


def get_phonology(words,
                  word_key = 'word',
                  phonemes_key = 'phonemes'):
    """
    """
    phonology = []
    for word in words:
        try:
            phonology.append({word_key: word,
                              phonemes_key: cmudict.dict()[word][0]})
        except KeyError:
            phonology.append({word_key: word,
                              phonemes_key: None})

    return phonology


def get_phonemes_ordered(phonology,
                         phonemes_key = 'phonemes'):
    """
    Gets a list of all phonemes in the lexicon
    """
    phonemes_ordered = []

    for word in phonology:
        try:
            phonemes_ordered += word[phonemes_key]
        except TypeError:
            continue

    return list(dict.fromkeys(phonemes_ordered).keys()) # As of Python 3.7, dict is guaranteed to preserve order


def convert_word_to_onehot(word_phonology, phonemes_ordered):
    """
    Append the series of one hot phoneme representations into a time series
    """
    phonemes_onehot = []

    try:
        phonemes_onehot = np.eye(len(phonemes_ordered))[[phonemes_ordered.index(phoneme) for phoneme in word_phonology]]

    except TypeError:
        return None

    return np.stack(phonemes_onehot, axis = 0).tolist()


def add_onehot_to_phonology(phonology,
                            phonemes_ordered = None,
                            phonemes_key = 'phonemes',
                            phonemes_onehot_key = 'phonemes_onehot'):
    """
    """
    if not phonemes_ordered:
        phonemes_ordered = get_phonemes_ordered(phonology = phonology,
                                                phonemes_key = phonemes_key)

    for i, _ in enumerate(phonology):
        phonology[i][phonemes_onehot_key] = convert_word_to_onehot(word_phonology = phonology[i][phonemes_key],
                                                                   phonemes_ordered = phonemes_ordered)

    return phonology


if __name__ == '__main__':
    lexicon_filename = 'representations/lexicon.csv'
    lexicon = pd.read_csv(lexicon_filename)

    phonology = get_phonology(words = list(lexicon['word']))

    # Now handled by the language interface
    """
    phonemes_ordered = get_phonemes_ordered(phonology = phonology)
    phonology = add_onehot_to_phonology(phonology = phonology, phonemes_ordered = phonemes_ordered)
    """

    phonology_filename_csv = 'representations/phonology_real.csv'
    pd.DataFrame(phonology).to_csv(phonology_filename_csv, index = False)