import pandas as pd

from pathlib import Path


def get_word_frequencies(utterances,
                         sep = ' '):
    """
    """
    counter = {}

    for utterance in utterances:
        for word in utterance.split(sep):
            if word not in counter:
                counter[word] = 1
            else:
                counter[word] += 1

    return counter

def get_word_probabilities(utterances, probabilities,
                           sep = ' '):
    """
    """
    counter = {}

    for utterance, probability in zip(utterances, probabilities):
        for word in utterance.split(sep):
            if word not in counter:
                counter[word] = 1 * probability
            else:
                counter[word] += 1 * probability

    return counter

def get_n_grams(utterances):
    """
    """
    pass

if __name__ == '__main__':
    language = 'datives'
    corpus = 'all_sentences.tsv'
    corpus_path = Path.cwd().joinpath(f'{language}/{corpus}')

    corpus = pd.read_csv(corpus_path, sep = '\t')
    print(corpus['sentence_str'])

    word_counts = get_word_probabilities(utterances = corpus['sentence_str'], probabilities = corpus['probability'])
    print(word_counts)