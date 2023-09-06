import itertools
import math
import random

import pandas as pd

def generate_phonology(words,
                       pattern = ['C1', 'V1', 'C2'],
                       category_nums = {'C1' : 4, 'V1' : 4, 'C2' : 4}):
    """
    """
    # Ensure the number of words can be handled by the categories
    num_words = len(words)
    assert math.prod(category_nums.values()) >= num_words

    # Generate all possible sequences of phones
    all_possible_phone_sequences = generate_all_possible_phone_sequences(pattern = pattern, category_nums = category_nums)

    # Assign phone sequences to words
    df_word_phonology = []
    for word, phone_sequence in zip(words, all_possible_phone_sequences):
        word_phonology = {'word' : word, 'phonemes' : phone_sequence}
        df_word_phonology.append(word_phonology)

    return df_word_phonology


def generate_all_phones_in_category(category, num_phones_in_category):
    """
    """
    phones = []

    for num in range(num_phones_in_category):
        phones.append(f'{category}-{num}')

    return phones


def generate_all_possible_phone_sequences(pattern, category_nums,
                                          shuffle = True,
                                          seed = 9):
    """
    """
    all_phones = []

    for phone_category in pattern:
        all_phones.append(generate_all_phones_in_category(category = phone_category, num_phones_in_category = category_nums[phone_category]))

    phone_sequences = list(itertools.product(*all_phones))

    if shuffle:
        random.seed(seed)
        random.shuffle(phone_sequences)

    return [list(phone_sequence) for phone_sequence in phone_sequences]


def scramble_illegal_phonology(accepted_words,
                               pattern = ['C1', 'V1', 'C2'],
                               category_nums = {'C1' : 4, 'V1' : 4, 'C2' : 4},
                               reject_repeat_phones = True):
    """
    """
    all_phones = []

    for phone_category in pattern:
        all_phones += generate_all_phones_in_category(category = phone_category, num_phones_in_category = category_nums[phone_category])
    all_phones = [all_phones] * len(pattern)

    phone_sequences = [list(phone_sequence) for phone_sequence in list(itertools.product(*all_phones)) if list(phone_sequence) not in accepted_words]

    if reject_repeat_phones:
        phone_sequences = [phone_sequence for phone_sequence in phone_sequences if len(set(phone_sequence)) == len(phone_sequence)]

    return [{'word' : n, 'phonemes' : s} for n, s in enumerate(phone_sequences)]


def scramble_legal_phonology(accepted_words,
                             pattern = ['C1', 'V1', 'C2'],
                             category_nums = {'C1' : 4, 'V1' : 4, 'C2' : 4},
                             reject_repeat_phones = True,
                             seed = 9):
    """
    """
    all_phones = []

    for phone_category in pattern:
        all_phones.append(generate_all_phones_in_category(category = phone_category, num_phones_in_category = category_nums[phone_category]))
    phone_sequences = list(itertools.product(*all_phones))
    
    all_legal_nonwords = [word for word in phone_sequences if list(word) not in accepted_words]

    return [{'word' : n, 'phonemes' : nw} for n, nw in enumerate(all_legal_nonwords)]


if __name__ == '__main__':
    lexicon_filename = 'representations/lexicon.csv'
    lexicon = pd.read_csv(lexicon_filename)

    pattern = ['C1', 'V1', 'C2']
    category_nums = {'C1' : 4, 'V1' : 4, 'C2' : 4}

    phonology = generate_phonology(words = list(lexicon['word']),
                                   pattern = pattern,
                                   category_nums = category_nums)
    accepted_words = [p['phonemes'] for p in phonology]
    print(accepted_words)

    illegal_nonwords = scramble_illegal_phonology(accepted_words = accepted_words,
                                                  pattern = pattern,
                                                  category_nums = category_nums)
    legal_nonwords = scramble_legal_phonology(accepted_words = accepted_words,
                                              pattern = pattern,
                                              category_nums = category_nums)
    print([p['phones'] for p in legal_nonwords])

    phonology_filename_csv = 'representations/phonology_artificial.csv'
    pd.DataFrame(phonology).to_csv(phonology_filename_csv, index = False) 

    illegal_nonword_filename_csv = 'representations/phonology_artificial_illegal_nonwords.csv'
    pd.DataFrame(illegal_nonwords).to_csv(illegal_nonword_filename_csv, index = False)

    legal_nonword_filename_csv = 'representations/phonology_artificial_legal_nonwords.csv'
    pd.DataFrame(legal_nonwords).to_csv(legal_nonword_filename_csv, index = False)