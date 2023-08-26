import random

import pandas as pd

from pathlib import Path
from itertools import product

def read_words(word_path,
               word_key = 'word',
               exclude_words = ['blinked', 'fed', 'gave', 'lent', 'mailed', 'ran', 'sat', 'slept', 'ate', 'drank', 'took', 'borrowed']):
    """
    """
    nonword_df = pd.read_csv(word_path)

    return [word for word in list(nonword_df[word_key]) if word not in exclude_words]


def generate_all_lists(items, list_length):
    """
    Note this will take a very long time using all nonwords
    """
    items = [items] * list_length

    all_lists = product(*items)

    return list(all_lists)

def dfify_lists(all_lists):
    """
    """
    return [{'utterance_lst' : l, 'utterance_str' : ' '.join(l)} for l in all_lists]

def generate_n_random_lists(n, items, list_length,
                            lst_key = 'utterance_lst',
                            str_key = 'utterance_str',
                            seed = 9):
    """
    """
    random.seed(seed)

    random_lists = [{lst_key : random.choices(items, k = list_length)} for _ in range(n)]

    # Add utterance_str
    for l in random_lists:
        l[str_key] = ' '.join(map(str, l[lst_key]))

    return random_lists

if __name__ == '__main__':
    language = 'datives'
    word_path = Path.cwd().joinpath(f'{language}/lexicon/representations/phonology_artificial.csv')
    words = read_words(word_path = word_path)

    # Generate memory lists
    seed = 9
    list_length = 4
    num_lists = 500
    all_lists = generate_n_random_lists(n = num_lists, items = words, list_length = list_length, seed = seed)
    all_lists = pd.DataFrame(all_lists)

    # Save memory lists
    memory_list_path = Path.cwd().joinpath(f'{language}/noun_memory_lists_length{list_length}_n{num_lists}.tsv')
    print(memory_list_path)
    all_lists.to_csv(memory_list_path, sep = '\t', index = False)