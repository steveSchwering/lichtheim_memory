import spacy

import numpy as np
import pandas as pd

from pathlib import Path


def cosine_similarity(vec1, vec2):
    """
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_similarities(semantics, embeddings,
                     word_key = 'word'):
    """
    """
    print("-- Finding similarities between all words")

    similarities = []
    for word_1 in semantics[word_key]:
        for word_2 in semantics[word_key]:
            pairing = {'word_1': word_1,
                       'word_2': word_2,
                       'similarity': cosine_similarity(embeddings[word_1], embeddings[word_2])}
            similarities.append(pairing)

    return pd.DataFrame(similarities)


def get_localist_similarities(semantics,
                              nonfeature_cols = ['word', 'pos'],
                              word_key = 'word'):
    """
    """
    print("Artificial language")
    just_embeddings = semantics.drop(nonfeature_cols, axis = 1)

    embeddings = {}
    for embedding, word in zip(just_embeddings.values.tolist(), semantics[word_key]):
        embeddings[word] = np.array(embedding)

    return get_similarities(semantics, embeddings, word_key)


def get_spacy_embeddings(semantics,
                         word_key = 'word'):
    """
    """
    print("-- Reading in spacy model")
    nlp = spacy.load("en_core_web_md")

    print("-- Extracting embeddings for words")
    embeddings = {}
    for word in semantics[word_key]:
        embeddings[word] = nlp(str(word)).vector

    return embeddings


def get_spacy_similarities(semantics,
                           word_key = 'word'):
    """
    """
    print("Spacy")

    embeddings = get_spacy_embeddings(semantics = semantics,
                                      word_key = word_key)

    return get_similarities(semantics, embeddings, word_key)


if __name__ == '__main__':
    semantics_file = Path.cwd().joinpath('representations/semantics_localist.csv')
    localist_similarities_file = Path.cwd().joinpath('similarities/localist_similarities.csv')
    spacy_similarities_file = Path.cwd().joinpath('similarities/spacy_similarities.csv')

    localist_representations = pd.read_csv(semantics_file)

    localist_similarities = get_localist_similarities(semantics = localist_representations)
    spacy_similarities = get_spacy_similarities(semantics = localist_representations)

    localist_similarities.to_csv(localist_similarities_file, index = False)
    spacy_similarities.to_csv(spacy_similarities_file, index = False)