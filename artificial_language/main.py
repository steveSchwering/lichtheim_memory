import random

import Structure
import Sentence
import Producer
import SGInterface
import LMInterface

import helperfunctions as hf

from pathlib import Path
from collections import Counter

def count_sentence_categories(sentence_sample, io_sample):
    """
    """
    sentence_structs = [sentence.structure for sentence in sentence_sample]
    sentence_struct_count = Counter(sentence_structs)
    print(sentence_struct_count)

    io_structs = [sentence['structure'] for sentence in io_sample]
    io_struct_count = Counter(io_structs)
    print(io_struct_count)

if __name__ == '__main__':
    language_name = 'datives'
    language_dir = Path.cwd().joinpath(f'language/{language_name}')

    # Create model interface
    interface_sg = SGInterface.SGInterface(language_dir = language_dir, phonology_filestem = 'phonology_artificial')
    interface_lm = LMInterface.LMInterface(language_dir = language_dir, phonology_filestem = 'phonology_artificial')

    # Generate sentences
    structures_path = language_dir.joinpath(f'grammar/structures')
    producer = Producer.Producer(structures_info_path = structures_path)
    sentences = Sentence.convert_sentence_dicts_to_objs(producer.generate_all_sentences())
    save_file = language_dir.joinpath(f'all_sentences.tsv')
    Sentence.save_sentences_info(sentences, save_file)

    # Test reading all sentences from file
    # THE BELOW IS ALL YOU NEED ONCE THE SENTENCES ARE GENERATED
    r_sentences = LMInterface.sentences_from_file(save_file)

    # Generate input/output patterns for all sentences
    examples = []
    for sentence in r_sentences:
        examples += interface_lm.generate_weighted_tasks_for_utterance(utterance = sentence['sentence_lst'], info = sentence)

    print(examples[0])