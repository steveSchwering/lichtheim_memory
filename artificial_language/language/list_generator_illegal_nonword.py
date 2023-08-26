import random

import pandas as pd

from pathlib import Path
from itertools import product

def read_nonwords(nonword_path,
                  nonword_key = 'word'):
    """
    """
    nonword_df = pd.read_csv(nonword_path)

    return list(nonword_df[nonword_key])


def generate_all_lists(items, list_length):
    """
    Note this will take a very long time using all nonwords
    """
    items = [items] * list_length

    all_lists = product(*items)

    return list(all_lists)


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
    nonword_path = Path.cwd().joinpath(f'{language}/lexicon/representations/phonology_artificial_illegal_nonwords.csv')
    nonwords = read_nonwords(nonword_path = nonword_path)

    # Generate memory lists
    seed = 9
    num_lists = 500
    list_length = 4
    memory_lists = generate_n_random_lists(n = num_lists, items = nonwords, list_length = list_length, seed = seed)
    memory_lists = pd.DataFrame(memory_lists)
    
    # Save memory lists
    memory_list_path = Path.cwd().joinpath(f'language/{language}/nonword_memory_lists_illegal_length{list_length}_n{num_lists}.tsv')
    memory_lists.to_csv(memory_list_path, sep = '\t', index = False)