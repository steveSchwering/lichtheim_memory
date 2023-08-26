import numpy as np
import pandas as pd
import helperfunctions as hf

from pathlib import Path


class SGInterface():
    def __init__(self, language_dir,
                 phonology_filestem = 'phonology_artificial'):
        self.language_dir = language_dir

        # Get general language information
        lexicon, phonology, semantics, roles = self._get_lexical_information(language_dir = language_dir, phonology_filestem = phonology_filestem)
        self.word_to_semantics = hf._convert_df_to_dict(df = semantics, key_col = 'word')
        self.role_to_index = hf._convert_list_to_elem_num_dict(list(roles['role']))

        # Generate interpreters for input and output
        self.gi_to_index, self.index_to_gi, self.pt_to_index, self.index_to_pt = self._construct_language_interface(lexicon, roles, semantics)

    def _get_lexical_information(self, language_dir,
                                 phonology_filestem = 'phonology_artificial',
                                 semantics_filestem = 'semantics_localist'):
        """
        Generates lexical, phonological, semantic, and role information
        """
        # Lexicon
        lexicon_path = Path.cwd().joinpath(language_dir).joinpath('lexicon/representations/lexicon.csv')
        lexicon = pd.read_csv(lexicon_path)

        # Phonology
        phonology_path = Path.cwd().joinpath(language_dir).joinpath(f'lexicon/representations/{phonology_filestem}.csv')
        phonology = pd.read_csv(phonology_path)

        # Semantics
        semantics_path = Path.cwd().joinpath(language_dir).joinpath(f'lexicon/representations/{semantics_filestem}.csv')
        semantics = pd.read_csv(semantics_path)

        # Roles
        roles_path = Path.cwd().joinpath(language_dir).joinpath('grammar/roles.csv')
        roles = pd.read_csv(roles_path)

        return lexicon, phonology, semantics, roles

    def _construct_language_interface(self, lexicon, roles, semantics,
                                      lexeme_key = 'word',
                                      time_dimensions = ['before_verb', 'verb', 'post_verbal'],
                                      role_key = 'role',
                                      semantics_drop_cols = ['word']):
        """
        Generates information necessary for the sentence gestalt model to interface with the language
        """
        # Generate dictionaries for interpreting sentence gestalt input
        # -- Composed of lexeme units plus time units
        word_to_index = hf._convert_list_to_elem_num_dict(l = lexicon[lexeme_key]) # Get dictionary of lexemes
        gi_to_index, index_to_gi = hf._add_elems_to_elem_num_dict(old_elem_to_index = word_to_index, new_elems = time_dimensions) # Add time dimension to one-hot word input

        # Generate dictionaries for interpreting probe/target input and output
        # -- Composed of role units plus semantic feature units
        role_to_index = hf._convert_list_to_elem_num_dict(l = roles[role_key])
        pt_to_index, index_to_pt = hf._add_elems_to_elem_num_dict(old_elem_to_index = role_to_index, new_elems = list(semantics.drop(columns = semantics_drop_cols).columns))

        return gi_to_index, index_to_gi, pt_to_index, index_to_pt

    # Sentence input to the Sentence Gestalt model
    def _get_gi_pattern_timing_signal(self, sentence_role_to_filler):
        """
        """
        flag_before_verb = True

        timing = []
        for role, word in sentence_role_to_filler.items():
            if role == 'verb':
                timing.append('verb')
                flag_before_verb = False

            if flag_before_verb:
                timing.append('before_verb')

            else:
                timing.append('post_verbal')

        return timing

    def _construct_gi_pattern(self, sentence_role_to_filler):
        """
        Converts sentence list to a tensor of one-hot word inputs
        """
        # Generate word inputs -- one-hot representations of words input to model
        gi_pattern = [self.gi_to_index[word] for role, word in sentence_role_to_filler.items()]
        gi_pattern = np.eye(len(self.gi_to_index))[gi_pattern]

        # Generate timing signal inputs -- turn on the timing signal units
        for step_gi_pattern, timing_signal in zip(gi_pattern, self._get_gi_pattern_timing_signal(sentence_role_to_filler)):
            step_gi_pattern[self.gi_to_index[timing_signal]] = 1

        return gi_pattern


    # Probing the Sentence Gestalt model: role and filler
    ### Generate a probe and target for a single role/filler
    def _construct_role_filler_vectors(self, role, filler):
        """
        Takes role and filler and generates a role vector and a filler vector
        """
        role_vector = np.eye(len(self.role_to_index))[self.pt_to_index[role]]
        role_empty = np.zeros(len(role_vector))

        filler_vector = self.word_to_semantics[filler]
        filler_empty = np.zeros(len(filler_vector))

        role_vector = np.concatenate((role_vector, filler_empty), axis = 0)
        filler_vector = np.concatenate((role_empty, filler_vector), axis = 0)

        return role_vector, filler_vector

    def _construct_role_filler_tensors(self, role, filler, sentence_len):
        """
        Takes a role and filler and converts them into a series of role/filler vectors
        Extends role and filler through time for the sentence gestalt model
        Note: the role and filler vectors are the same at each time step, so we can just copy it for the length of the sentence
        """
        # Create role filler vector for 1 time step
        role_vector, filler_vector = self._construct_role_filler_vectors(role = role,
                                                                         filler = filler)

        # Extend role and filler vectors for time steps equal to length of sentence
        role_tensor = np.tile(role_vector, (sentence_len, 1))
        filler_tensor = np.tile(filler_vector, (sentence_len, 1))

        return role_tensor, filler_tensor

    def _assign_probe_and_target(self, probe_type, role_tensor, filler_tensor):
        """
        Returns probe and target for the sentence gestalt model
        + If probe is 'role', role_tensor is returned first
        + If probe is 'filler', filler tensor is return first
        """
        if probe_type == 'role':
            return role_tensor, filler_tensor

        elif probe_type == 'filler':
            return filler_tensor, role_tensor

    def _construct_probe_and_target(self, probe_type, role, filler, sentence_len):
        """
        Returns probe and target for a given type of probe and role/filler pair for the sentence gestalt model
        """
        role_tensor, filler_tensor = self._construct_role_filler_tensors(role = role,
                                                                         filler = filler,
                                                                         sentence_len = sentence_len)

        probe, target = self._assign_probe_and_target(probe_type = probe_type,
                                                      role_tensor = role_tensor,
                                                      filler_tensor = filler_tensor)

        return probe, target

    def _construct_all_pt_patterns(self, sentence_role_to_filler,
                                   probe_types = ['role', 'filler']):
        """
        Returns all probes and targets as dictionaries tracking probe_type, role, filler, probe tensor, and target tensor
        """
        pt_patterns = []

        for role, filler in sentence_role_to_filler.items():
            for probe_type in probe_types:
                probe, target = self._construct_probe_and_target(probe_type = probe_type,
                                                                 role = role,
                                                                 filler = filler,
                                                                 sentence_len = len(sentence_role_to_filler))

                pt_patterns.append({'probe_type': probe_type,
                                    'role': role,
                                    'filler': filler,
                                    'probe': probe,
                                    'target': target})

        return pt_patterns

    def construct_all_io_patterns(self, sentence_role_to_filler):
        """
        Generates the input to the sentence gestalt and the probes and targets
        """
        # Create gestalt input
        gi_pattern = self._construct_gi_pattern(sentence_role_to_filler = sentence_role_to_filler)

        # Create all probes and targets
        pt_patterns = self._construct_all_pt_patterns(sentence_role_to_filler = sentence_role_to_filler)

        return gi_pattern, pt_patterns

    def batch_sentence_io_examples(self, gi_pattern, pt_patterns,
                                   time_dim = 0,
                                   batch_dim = 1):
        """
        Batch all examples for a sentence into one -- assumes all pt_patterns of same length
        """
        # Batch together N gi_patterns where N is the number of pt_patterns
        gestalts = [gi_pattern for _ in pt_patterns]
        gi_pattern_batched = np.stack(gestalts, axis = batch_dim)
        
        # Batch together all pt_patterns
        probes = [pt_info['probe'] for pt_info in pt_patterns]
        probe_batched = np.stack(probes, axis = batch_dim)
        targets = [pt_info['target'] for pt_info in pt_patterns]
        target_batched = np.stack(targets, axis = batch_dim)
        
        probe_types = [pt_info['probe_type'] for pt_info in pt_patterns]
        roles = [pt_info['role'] for pt_info in pt_patterns]
        fillers = [pt_info['filler'] for pt_info in pt_patterns]

        # Combine this all into a neat package
        io = {'gi' : gi_pattern_batched,
              'probe' : probe_batched,
              'target' : target_batched,
              'probe_type' : probe_types,
              'role' : roles,
              'filler' : fillers}

        return io

    # Filter and package SG input and output, so one example contains all information
    def _filter_pt_patterns(self, pt_patterns,
                            filter_probe = None,
                            filter_role = None,
                            filter_filler = None):
        """
        Takes a list of dictionaries containing information about probes and targets. Filters these examples
        to find match on probe_type, role, and filler
        """
        filtered_pt_patterns = []

        for example in pt_patterns:
            if filter_probe:
                if example['probe_type'] not in filter_probe:
                    continue
            if filter_role:
                if example['role'] not in filter_role:
                    continue
            if filter_filler:
                if example['filler'] not in filter_filler:
                    continue
            filtered_pt_patterns.append(example)

        return filtered_pt_patterns

    def package_io(self, sentence_info, gi_pattern, pt_patterns,
                   add_adjusted_probability = True):
        """
        Helper function that packs I/O to be used by Sentece Gestalt

        sentence_info is a dictionary of sentence information
        """
        packaged_info_and_patterns = []

        for example in pt_patterns:
            example.update({'gi_pattern': gi_pattern})
            example.update(sentence_info)
            if add_adjusted_probability:
                example['length_adjusted_probability'] = example['probability'] / len(example['sentence_lst'])
            packaged_info_and_patterns.append(example)

        return packaged_info_and_patterns

    def package_batched_io(self, sentence_info, io):
        """
        """
        io.update(sentence_info)
        return io


if __name__ == '__main__':
    language_name = 'datives'
    language_dir = f'language/{language_name}'

    interface = SGInterface(language_dir = language_dir)
    
    sentence_info = {'role_to_filler' : {'subject' : 'man', 'verb' : 'gave', 'indirect_object' : 'boy', 'direct_object' : 'tea'},
                     'probability' : 0.50,
                     'sentence_lst' : [{'subject' : 'man'}, {'verb' : 'gave'}, {'indirect_object' : 'boy'}, {'direct_object' : 'tea'}]}

    gi_pattern, pt_patterns = interface.construct_all_io_patterns(sentence_role_to_filler = sentence_info['role_to_filler'])
    io = interface.batch_sentence_io_examples(gi_pattern = gi_pattern, pt_patterns = pt_patterns)
    packaged_io = interface.package_batched_io(sentence_info = sentence_info, io = io)
    print(packaged_io['filler'])
    for t_targ, word in zip(packaged_io['target'], packaged_io['sentence_lst']):
        print(f'{word} -- target output array of shape {t_targ.shape}\n{t_targ}')