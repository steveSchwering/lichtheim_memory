import itertools

import pandas as pd
import helperfunctions as hf

from pathlib import Path

class Structure():
    def __init__(self, structure, structure_path,
                 structure_prob = 1.0,
                 needed_roles = None):
        """
        Class tracking generic structure information

        conditional_filler_probs in form of verb : role : filler : prob
        """
        self.structure = structure
        self.structure_path = structure_path
        assert self.structure_path.exists() & self.structure_path.is_dir()

        self.structure_prob = structure_prob

        if needed_roles:
            self.needed_roles = needed_roles
        else:
            self.needed_roles = self._read_needed_roles()

        self.verb_probs = self._read_verb_probs()
        self.conditional_filler_probs = self._read_conditional_filler_probs()

    def _read_needed_roles(self, delimiter = ' '):
        """
        Identifies roles needed for structure from file
        """
        needed_roles_path = self.structure_path.joinpath('needed_roles.txt')

        with open(needed_roles_path, 'r') as f:
            roles = f.readline().split(delimiter)

        return roles

    def _read_verb_probs(self,
                         verbs_col = 'verb',
                         weight_col = 'weight'):
        """
        Identifies probability of verb conditioned on structure
        """
        verb_probs_path = self.structure_path.joinpath(f'role_filler_probabilities/condition_verb/verb_weights.csv')
        verb_probs = pd.read_csv(verb_probs_path)
        
        return dict(zip(verb_probs[verbs_col], hf._convert_weights_to_probs(weights = verb_probs[weight_col])))

    def _read_verb_conditional_filler_probs(self, verb_cond_prob_path,
                                            filler_key = 'filler',
                                            weight_key = 'weight',
                                            probability_key = 'probability'):
        """
        Identifies probability of fillers conditioned on structure and verb

        + Requires verb_cond_prob_path to specify condition of verb
        """
        role_search = verb_cond_prob_path.glob(f'*.csv')

        role_filler_weights = {}
        for role_path in role_search:
            filler_weights = pd.read_csv(role_path)
            filler_weights[probability_key] = hf._convert_weights_to_probs(weights = filler_weights[weight_key])
            role_filler_weights[role_path.stem] = dict(zip(filler_weights[filler_key], filler_weights[probability_key]))

        return role_filler_weights

    def _read_conditional_filler_probs(self):
        """
        Wrapper around _read_verb_conditional_filler_probs to apply to all verbs
        """
        verb_cond_prob_path_search = self.structure_path.joinpath('role_filler_probabilities/condition_verb').glob('*')
        verb_cond_prob_paths = [vf for vf in verb_cond_prob_path_search if vf.is_dir()]

        verb_role_fillers = {}
        for verb_cond_prob_path in verb_cond_prob_paths:
            verb = verb_cond_prob_path.stem
            verb_role_fillers[verb] = self._read_verb_conditional_filler_probs(verb_cond_prob_path = verb_cond_prob_path)

        return verb_role_fillers

    # Generate sentences from Structure
    # -- Generate a single sentence
    def generate_random_sentence(self, verb = None):
        """
        Generates random sentence 
        """
        if not verb:
            verb, verb_prob = hf._choose_from_probability_dict(probability_dict = self.verb_probs)
        else:
            verb_prob = self.verb_probs[verb]

        # Iterate through needed roles and select fillers
        fillers = []
        role_to_filler = {}
        probability = self.structure_prob * verb_prob
        for role in self.needed_roles:
            filler, prob = hf._choose_from_probability_dict(probability_dict = self.conditional_filler_probs[verb][role])
            fillers.append(filler)
            role_to_filler[role] = filler
            probability = probability * prob

        # Assemble sentence
        sentence = {'structure' : self.structure,
                    'needed_roles' : self.needed_roles,
                    'role_to_filler' : role_to_filler,
                    'probability' : probability}

        return sentence

    # -- Generate all possible sentences
    def _filter_fillers(self, filler_probs,
                        probability_threshold = 0.0):
        """
        Filters fillers to ensure that probability of filler is greater than threshold
        """
        return [(filler, prob) for (filler, prob) in filler_probs.items() if prob > probability_threshold]

    def _get_all_possible_role_fillers(self, verb,
                                       probability_threshold = 0.0):
        """
        Returns possible fillers for role given verb and probability_threshold
        """
        possible_role_fillers = []
        for role in self.needed_roles:
            possible_role_fillers.append(self._filter_fillers(filler_probs = self.conditional_filler_probs[verb][role], 
                                                              probability_threshold = probability_threshold))

        return possible_role_fillers

    def _convert_sentence_filler_tuples_to_prob(self, sentence_filler_tuples,
                                                condition_probability = 1.0):
        """
        Calculates probability of sentences by multiplyinf together probability of fillers with passed probability
        """
        sentence_probabilities = []
        for sentence_tuple in sentence_filler_tuples:

            probability = 1.0 * condition_probability
            for filler_tuple in sentence_tuple:
                probability = probability * filler_tuple[1]
            sentence_probabilities.append(probability)

        return sentence_probabilities

    def generate_all_sentences(self):
        """
        Returns dictionary of structure, role_to_filler, needed_roles, and probability for each possible sentence
        """
        sentences_info = []
        total_probability = 0

        for (verb, verb_prob) in self.verb_probs.items():
            if verb_prob <= 0.0:
                continue

            # Get all fillers that have a non-zero probability conditioned on verb
            possible_role_fillers = self._get_all_possible_role_fillers(verb = verb)

            # Generate all possible sentences and sentence probabilities from list of available fillers
            possible_sentence_probability_tuples = list(itertools.product(*possible_role_fillers))
            
            # Generate sentences for each combination
            for sentence_tuple in possible_sentence_probability_tuples:
                try:
                    words, probabilities = zip(*sentence_tuple)
                    probability = (self.structure_prob * self.verb_probs[verb] * hf.prod(probabilities))
                    sentences_info.append({'structure' : self.structure,
                                           'role_to_filler' : dict(zip(self.needed_roles, words)),
                                           'needed_roles' : self.needed_roles,
                                           'probability' : probability})
                    total_probability += probability
                except ValueError:
                    continue

        return sentences_info

def read_structures(structures_info_path,
                    structure_key = 'structure',
                    weight_key = 'weight'):
    """
    """
    structure_weights_path = structures_info_path.joinpath('structure_weights.csv')

    structure_weights_df = pd.read_csv(structure_weights_path)
    structure_weights = dict(zip(structure_weights_df[structure_key], hf._convert_weights_to_probs(structure_weights_df[weight_key])))
    
    structure_search = [structure for structure in structures_info_path.glob('*') if structure.is_dir()]

    structures = []
    for structure_path in structure_search:
        structures.append(Structure(structure = structure_path.stem,
                                    structure_path = structure_path,
                                    structure_prob = structure_weights[structure_path.stem]))

    return structures


if __name__ == '__main__':
    structures_path = Path.cwd().joinpath('language/datives/grammar/structures')
    structures = read_structures(structures_info_path = structures_path)

    sentences = []
    for structure in structures:
        sentences += structure.generate_all_sentences()

    print(sentences)