import random
import uuid
import ast

import numpy as np

from pathlib import Path


# Cnte Sentence objects
class Sentence():
    def __init__(self,
                 structure = None,
                 needed_roles = [],
                 sentence_id = None,
                 role_to_filler = {},
                 probability = 0,
                 sentence_gestalt = None):
        """
        Store information about the sentence type and the fillers of the sentence
        """
        # Set info about sentence
        self.structure = structure
        self.needed_roles = needed_roles

        # Define roles and fillers
        self.role_to_filler = {}
        self.filler_to_role = {}
        self.set_role_filler(role_to_filler = role_to_filler)

        # Track probability of sentence
        self.probability = probability

        # Construct sentence
        # Generates: self.sentence_str and self.sentence_lst
        self.sentence_str = None
        self.sentence_lst = None
        self.sentence_role_filler_lst = None
        self.construct_sentence()

        # Sentence semantic representation
        self.sentence_gestalt = None

        # Set ID
        if sentence_id:
            self.sentence_id = sentence_id
        else:
            self.sentence_id = self.generate_random_id()

    # Store information about the sentence
    def generate_random_id(self):
        """
        """
        return f'{self.structure}_{self.role_to_filler["verb"]}_{uuid.uuid4()}'

    def set_sentence_id(self, sentence_id):
        """
        """
        self.sentence_id = sentence_id

    def set_role_filler(self, role_to_filler):
        """
        Sets an filler in the filler dictionary
        """
        self.role_to_filler.update(role_to_filler)
        self.filler_to_role.update({v: k for k, v in role_to_filler.items()})
        self.construct_sentence()

    def set_probability(self, probability):
        """
        """
        self.probability = probability

    def construct_sentence(self,
                           sentence_str_splitter = ' '):
        """
        Returns sentence string as well as list of sentence fillers
        """
        self.sentence_lst = [self.role_to_filler[elem] for elem in self.needed_roles if elem in self.role_to_filler.keys()]
        self.sentence_str = sentence_str_splitter.join(self.sentence_lst)
        self.sentence_role_filler_lst = [{elem: self.role_to_filler[elem]} for elem in self.needed_roles if elem in self.role_to_filler.keys()]

    def set_sentence_gestalt(self, sentence_gestalt = None):
        """
        Sets sentence gestalt representation to numpy array of real-valued sentence representation
        """
        self.sentence_gestalt = sentence_gestalt

    # Get information about the sentence
    def get_needed_roles(self,
                         exclude_verb = True):
        """
        Returns needed fillers for the sentence type excluding verb by default
        """
        if exclude_verb:
            return [role for role in self.needed_roles if role != 'verb']
        return self.needed_roles

    def get_role_filler_pairs(self):
        """
        Returns list of tuples of all role/filler pairs in the model e.g. [('subj', 'man'), ('verb', 'slept')]
        """
        return list(self.role_to_filler.items())

    def package_sentence_info(self):
        """
        """
        return {'sentence_id': self.sentence_id,
                'structure': self.structure,
                'needed_roles': self.needed_roles,
                'sentence_lst': self.sentence_lst,
                'sentence_str': self.sentence_str,
                'role_to_filler': self.role_to_filler,
                'filler_to_role': self.filler_to_role,
                'probability': self.probability,
                'verb': self.role_to_filler['verb']}

    # Generate components of the Lichtheim-memory model input/output
    def _sentence_to_phonology(self, word_to_phonology_map):
        """
        """
        sentence_phonology =[]
        for word in self.sentence_lst:
            word_phonology = word_to_phonology_map[word]
            sentence_phonology.append(word_phonology)

        return sentence_phonology

    def construct_lm_comprehension_io(self, word_to_phonology_map):
        """
        Input is series of phonemes
        Output is sentence semantic information
        """
        sentence_phonology = self._sentence_to_phonology(word_to_phonology_map = word_to_phonology_map)

        sentence_semantic_output = len(sentence_phonology) * self.sentence_gestalt


        return {'phonology_input': sentence_phonology,
                'sentence_semantic_output': sentence_semantic_output}

    def construct_lm_production_io(self, word_to_phonology_map):
        """
        Input is sentence semantic information
        Output is series of phonemes
        """
        sentence_phonology = self._sentence_to_phonology(word_to_phonology_map = word_to_phonology_map)

        sentence_semantic_input = len(sentence_phonology) * self.sentence_gestalt

        return {'sentence_semantic_input': sentence_semantic_input,
                'phonology_output': sentence_phonology}


    def construct_lm_repetition_io(self, word_to_phonology_map):
        """
        Input is series of phonemes
        Output is series of phonemes
        """
        sentence_phonology = self._sentence_to_phonology(word_to_phonology_map = word_to_phonology_map)

        zeroed_phonology = np.zeros(sentence_phonology.shape)

        return {'phonology_input': np.concatenate((sentence_phonology, zeroed_phonology), axis = 0),
                'phonology_output': np.concatenate((zeroed_phonology, sentence_phonology), axis = 0)}

    def construct_all_lm_io_patterns(self, word_to_phonology_map):
        """
        Returns comprehension, production, and repetition dictionaries
        """
        comprehension = self.construct_lm_comprehension_io(word_to_phonology_map = word_to_phonology_map)
        production = self.construct_lm_production_io(word_to_phonology_map = word_to_phonology_map)
        repetition = self.construct_lm_repetition_io(word_to_phonology_map = word_to_phonology_map)

        return comprehension, production, repetition

# Specific Sentence objects
class Intransitive(Sentence):
    def __init__(self,
                 role_to_filler = {},
                 needed_roles = ['subject', 'verb']):
        """
        Instantiates an intransitive sentence
        """
        Sentence.__init__(self, role_to_filler = role_to_filler, needed_roles = needed_roles)
        self.structure = 'intransitive'


class Transitive(Sentence):
    def __init__(self,
                 role_to_filler = {},
                 needed_roles = ['subject', 'verb', 'direct_object']):
        """
        Instantiates a transitive sentence
        """
        Sentence.__init__(self, role_to_filler = role_to_filler, needed_roles = needed_roles)
        self.structure = 'transitive'


class Ditransitive(Sentence):
    def __init__(self,
                 fillers = {},
                 needed_roles = ['subject', 'verb', 'indirect_object', 'direct_object']):
        """
        Instantiates a ditransitive sentence
        """
        Sentence.__init__(self, role_to_filler = fillers, needed_roles = needed_roles)
        self.structure = 'ditransitive'


def create_empty_sentence(structure):
    """
    """
    if structure == 'intransitive':
        sentence = Intransitive()
    elif structure == 'transitive':
        sentence = Transitive()
    elif structure == 'ditransitive':
        sentence = Ditransitive()

    return sentence


def convert_sentence_dict_to_obj(sentence_dict):
    """
    """
    try:
        return Sentence(sentence_id = sentence_dict['sentence_id'],
                        structure = sentence_dict['structure'],
                        role_to_filler = sentence_dict['role_to_filler'],
                        needed_roles = sentence_dict['needed_roles'],
                        probability = sentence_dict['probability'])
    except KeyError:
        return Sentence(structure = sentence_dict['structure'],
                        role_to_filler = sentence_dict['role_to_filler'],
                        needed_roles = sentence_dict['needed_roles'],
                        probability = sentence_dict['probability'])


def convert_sentence_dicts_to_objs(sentence_dicts):
    """
    """
    return [convert_sentence_dict_to_obj(sent) for sent in sentence_dicts]


def save_sentences_info(sentences, save_file,
                        header = ['sentence_id', 'structure', 'needed_roles', 'sentence_lst', 'sentence_str', 'role_to_filler', 'filler_to_role', 'probability', 'verb'],
                        delimiter = '\t',
                        newline = '\n',
                        write_command = 'w',
                        write_header = True):
    """
    Saves information about sentences
    """
    with open(save_file, write_command) as f:
        if write_header:
            f.write(delimiter.join(header) + newline)

        for sentence_num, sentence in enumerate(sentences):
            write_dct = sentence.package_sentence_info()
            write_lst = [str(write_dct[var]) for var in header]
            write_str = delimiter.join(write_lst) + newline
            f.write(write_str)


if __name__ == '__main__':
    structure = 'ditransitive'
    role_to_filler = {'subject': 'dog',
                      'verb' : 'gave',
                      'indirect_object' : 'man',
                      'direct_object' : 'ball'}
    needed_roles = ['subject', 'verb', 'indirect_object', 'direct_object']
    probability = 0.5

    sent = convert_sentence_dict_to_obj(sentence_dict = {'structure' : structure,
                                                         'role_to_filler' : role_to_filler,
                                                         'needed_roles' : needed_roles,
                                                         'probability' : probability})
    print(sent.sentence_id)
    print(sent.sentence_str)
    print(sent.sentence_lst)
    print(sent.sentence_role_filler_lst)