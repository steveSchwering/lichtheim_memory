import ast

import pandas as pd

from pathlib import Path
from itertools import permutations

def read_sentences(language_path,
                   sentence_filename = 'all_sentences.tsv',
                   sep = '\t',
                   sentence_col = 'sentence_lst'):
    """
    """
    all_sentences_filename = language_path.joinpath(sentence_filename)

    sentences_df = pd.read_csv(all_sentences_filename, sep = sep)

    sentences = sentences_df[sentence_col]

    sentences = [ast.literal_eval(s) for s in sentences]

    return sentences

def get_bigrams(utterance):
    """
    """
    bigrams = []

    for elem1, elem2 in zip(utterance, utterance[1:]):
        bigrams.append(tuple([elem1, elem2]))

    return set(bigrams)

def get_bigrams_wrapper(utterances):
    """
    """
    all_bigrams = []

    for u in utterances:
        all_bigrams += get_bigrams(utterance = u)

    return set(all_bigrams)

def generate_combinations_and_check_bigrams(sentences, bigrams):
    """
    Finds all permutations of a sentence, removing those permutations that have a bigram in bigrams
    """
    valid_scrambled_sents = []

    for sentence in sentences:
        perm_sentences = permutations(sentence, len(sentence))

        for perm_sent in perm_sentences:
            bigrams_perm_sent = get_bigrams(utterance = perm_sent)
            if bigrams_perm_sent.isdisjoint(bigrams):
                valid_scrambled_sents.append(perm_sent)

    return valid_scrambled_sents

def save_scrambled_sentences(scrambled_sentences, language_path,
                              utterance_header = ['utterance_str', 'utterance_lst'],
                              random_list_filename = 'scrambled_sentences.tsv',
                              sep = '\t'):
    """
    """
    random_list_filename = language_path.joinpath(random_list_filename)

    with open(random_list_filename, 'w') as f:
        f.write(f'{sep.join(utterance_header)}\n')
        for rl in scrambled_sentences:
            f.write(f'{" ".join(rl)}{sep}{list(rl)}\n')

if __name__ == '__main__':
    # Get sentences
    language = 'datives'
    language_path = Path.cwd().joinpath(f'{language}')
    sentences = read_sentences(language_path = language_path)

    # Get bigrams
    bigrams = get_bigrams_wrapper(utterances = sentences)

    # Get all combinations
    scrambled_sentences = generate_combinations_and_check_bigrams(sentences = sentences, bigrams = bigrams)
    save_scrambled_sentences(scrambled_sentences = scrambled_sentences, language_path = language_path)