import ast

import pandas as pd

from pathlib import Path

def read_sentences(language_path,
                   sentence_filename = 'all_sentences.tsv',
                   sep = '\t',
                   sentence_col = 'sentence_lst',
                   structure = 'ditransitive'):
    """
    Reads in all legal sentences of the artificial language
    """
    all_sentences_filename = language_path.joinpath(sentence_filename)

    sentences_df = pd.read_csv(all_sentences_filename, sep = sep)
    sentences_df = sentences_df.loc[sentences_df['structure'] == structure] 

    sentences = sentences_df[sentence_col]

    sentences = [ast.literal_eval(s) for s in sentences]

    return sentences


def read_structure_verbs(language_path, structure,
                         structure_path = 'grammar/structures',
                         verb_bias_path = 'role_filler_probabilities/condition_verb/verb_weights.csv',
                         sep = ',',
                         minimum_weight = 1):
    """
    Reads in all verbs that will replace the verb in the target structure
    """
    verb_filename = language_path.joinpath(structure_path).joinpath(structure).joinpath(verb_bias_path)

    verbs_df = pd.read_csv(verb_filename, sep = sep)
    verbs_df = verbs_df.loc[verbs_df['weight'] >= minimum_weight]

    return list(verbs_df['verb'])


def read_inanimate_nouns(language_path,
                         lexicon_path = 'lexicon/representations/semantics_localist.csv',
                         sep = ',',
                         verb_filter = 'action',
                         animate_filter = 'person'):
    """
    Identifies inanimate nouns from the lexicon, first filtering out all "action" words (i.e. verbs),
    then filtering out the "person" words that are remaining
    """
    lexicon_filename = language_path.joinpath(lexicon_path)
    lexicon_df = pd.read_csv(lexicon_filename, sep = sep)

    # Filter
    lexicon_df = lexicon_df.loc[lexicon_df['action'] != 1]
    lexicon_df = lexicon_df.loc[lexicon_df['person'] != 1]

    return list(lexicon_df['word'])


def replace_position_with_alternative(sentences, position, alternatives):
    """
    """
    orig_sentences = []
    alt_sentences = []
    for orig_sent in sentences:
        for alt in alternatives:
            orig_sentences.append(orig_sent)
            alt_sent = orig_sent.copy()
            alt_sent[position] = alt
            alt_sentences.append(alt_sent)

    return alt_sentences, orig_sentences

def save_scrambled_sentences(scrambled_sentences, orig_sentences, language_path, filename,
                             utterance_header = ['utterance_str', 'utterance_lst', 'orig_utterance_str', 'orig_utterance_lst'],
                             sep = '\t'):
    """
    """
    filename = language_path.joinpath(filename)

    with open(filename, 'w') as f:
        f.write(f'{sep.join(utterance_header)}\n')
        for rl, ol in zip(scrambled_sentences, orig_sentences):
            f.write(f'{" ".join(rl)}{sep}{list(rl)}{sep}{" ".join(ol)}{sep}{list(ol)}\n')


if __name__ == '__main__':
    # Return all sentences of the appropriate type
    language = 'datives'
    language_path = Path.cwd().joinpath(f'{language}')
    sentences = read_sentences(language_path = language_path,
                               structure = 'ditransitive')

    # Read replacement verbs
    verbs_intransitive = read_structure_verbs(language_path = language_path,
                                              structure = 'intransitive')
    verbs_transitive = read_structure_verbs(language_path = language_path,
                                            structure = 'transitive',
                                            minimum_weight = 9)
    nouns_inanimate = read_inanimate_nouns(language_path = language_path)

    # Substitute verbs in ditransitive sentence with alternative verbs
    #-- Ditransitive verb -> intransitive verb
    dit_int_sents, dit_int_orig_sents = replace_position_with_alternative(sentences = sentences, position = 1, alternatives = verbs_intransitive)
    #-- Ditransitive verb -> transitive verb
    dit_trn_sents, dit_trn_orig_sents = replace_position_with_alternative(sentences = sentences, position = 1, alternatives = verbs_transitive)
    #- Ditransitive verb -> inanimate noun
    dit_inn_sents, dit_inn_orig_sents = replace_position_with_alternative(sentences = sentences, position = 1, alternatives = nouns_inanimate)
    #- Animate subject noun -> inanimate subject noun
    ansn_inn_sents, ansn_inn_orig_sents = replace_position_with_alternative(sentences = sentences, position = 0, alternatives = nouns_inanimate)
    # Postervabl animate noun -> inanimate subject noun
    pvan_inn_sents, pvan_inn_orig_sents = replace_position_with_alternative(sentences = sentences, position = 2, alternatives = nouns_inanimate)

    # Save sentences
    save_scrambled_sentences(scrambled_sentences = dit_int_sents, orig_sentences = dit_int_orig_sents,
                             language_path = language_path,
                             filename = 'sentence_vb_int.tsv')
    save_scrambled_sentences(scrambled_sentences = dit_trn_sents, orig_sentences = dit_trn_orig_sents,
                             language_path = language_path,
                             filename = 'sentence_vb_trn.tsv')
    save_scrambled_sentences(scrambled_sentences = dit_inn_sents, orig_sentences = dit_inn_orig_sents,
                             language_path = language_path,
                             filename = 'sentence_vb_inn.tsv')
    save_scrambled_sentences(scrambled_sentences = ansn_inn_sents, orig_sentences = ansn_inn_orig_sents,
                             language_path = language_path,
                             filename = 'sentence_snb_inn.tsv')
    save_scrambled_sentences(scrambled_sentences = pvan_inn_sents, orig_sentences = pvan_inn_orig_sents,
                             language_path = language_path,
                             filename = 'sentence_pvnb_inn.tsv')