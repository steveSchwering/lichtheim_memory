import ast
import itertools

import numpy as np
import pandas as pd
import helperfunctions as hf

from pathlib import Path


class LMInterface():
    def __init__(self, language_dir,
                 phonology_filestem = 'phonology'):
        """
        """
        self.language_dir = language_dir

        # Get general language infomration
        self.lexicon, self.phonology, self.lexical_semantics = self._get_lexical_information(language_dir = language_dir,
                                                                                             phonology_filestem = phonology_filestem)
        self.word_to_semantics = hf._convert_df_to_dict(df = self.lexical_semantics, key_col = 'word')
        self.word_to_phonology = hf._convert_cols_to_dict(df = self.phonology, key_col = 'word', value_col = 'phonemes')

        # Get sentence semantic information
        self.sentence_to_sg = self._get_issformation(language_dir = language_dir)

        # Get information about sentence semantics
        self.ss_io_size = self.sentence_to_sg[list(self.sentence_to_sg)[0]].shape[0]

        # Construct interface between language and model
        self.ph_to_index, self.index_to_ph, self.ws_to_index, self.index_to_ws = self._construct_language_interface()

    # Get lexical information
    def _get_lexical_information(self, language_dir,
                                 phonemes_key = 'phonemes',
                                 phonology_filestem = 'phonology_artifical',
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
        phonology[phonemes_key] = phonology[phonemes_key].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)

        # Semantics
        semantics_path = Path.cwd().joinpath(language_dir).joinpath(f'lexicon/representations/{semantics_filestem}.csv')
        semantics = pd.read_csv(semantics_path)

        return lexicon, phonology, semantics

    def _get_phonemes_ordered(self,
                              phonemes_key = 'phonemes',
                              phoneme_pad = '<END>'):
        """
        Gets a list of all phonemes of words in the language
        """
        phonemes_ordered = []

        phoneme_cols = [c for c in self.phonology.columns if phonemes_key in c]

        for key in phoneme_cols:
            for phones in self.phonology[key]:
                try:
                    phonemes_ordered += phones
                except TypeError:
                    continue
                except ValueError:
                    continue

        return list(dict.fromkeys(phonemes_ordered).keys()) # As of Python 3.7, dict is guaranteed to preserve order

    # Get sentence semantic information
    def _get_issformation(self, language_dir,
                            spacer = ' '):
        """
        """
        ss_path = Path.cwd().joinpath(language_dir).joinpath('sgs')
        
        sgs = {}
        for file in ss_path.glob('*.npy'):
            sentence_str = str(file.stem).replace('_', spacer)
            sgs[sentence_str] = np.load(file)

        return sgs

    # Interpreters
    def _construct_language_interface(self,
                                      semantics_drop_cols = ['word']):
        """
        Generates dictionaries to interpret phonology and word semantics IO. Note, sentence semantics
        does not have any human readable interpretation, as they are derived from SG, so no dictionary is
        generated for sentence semantics.
        """
        # Phonology
        phonemes_ordered = self._get_phonemes_ordered()
        ph_to_index = hf._convert_list_to_elem_num_dict(l = phonemes_ordered)
        index_to_ph = hf._reverse_dictionary(d = ph_to_index)

        # Lexical semantics
        ws_features = self.lexical_semantics.drop(columns = semantics_drop_cols).columns
        ws_to_index = hf._convert_list_to_elem_num_dict(l = ws_features)
        index_to_ws = hf._reverse_dictionary(d = ws_to_index)

        return ph_to_index, index_to_ph, ws_to_index, index_to_ws

    # Turn a word into model-readable format
    def _convert_word_to_ws_pattern(self, word):
        """
        Generates word semantic representation repeated through time for the number of phonemes in the word
        """
        lexical_semantics = self.word_to_semantics[word]

        return np.tile(lexical_semantics, (len(self.word_to_phonology[word]), 1))

    def _convert_phonology_to_ph_pattern(self, word_phonology):
        """
        Generates phonological representation of each phoneme for 1 time step
        """
        all_phonemes = np.eye(len(self.ph_to_index)) # Generate square matrix of all phonemes
        phonemes_onehot = all_phonemes[[self.ph_to_index[phoneme] for phoneme in word_phonology]] # Select phonemes in word

        return np.stack(phonemes_onehot, axis = 0)

    def _convert_word_to_ph_pattern(self, word):
        """
        Wraps ph_pattern generator for word string input
        """
        word_phonology = self.word_to_phonology[word]

        return self._convert_phonology_to_ph_pattern(word_phonology = word_phonology)

    # Turn a sentence into a model-readable format
    def _convert_sentence_to_ph_pattern(self, sentence_lst):
        """
        Wraps ph_pattern generator for sentence list input
        """
        return np.concatenate([self._convert_word_to_ph_pattern(word = word) for word in sentence_lst], axis = 0)

    # Generate I/O representations for tasks
    # -- Word
    def repetition_word(self, word):
        """
        """
        word_ph = self._convert_word_to_ph_pattern(word = word)
        silence = np.zeros(word_ph.shape)

        return {'task' : 'repetition_word',
                'iph' : np.concatenate((word_ph, silence), axis = 0),
                'oph_targ' : np.concatenate((silence, word_ph), axis = 0)}

    def comprehension_word(self, word,
                           time_dim = 0):
        """
        """
        iph = self._convert_word_to_ph_pattern(word = word)
        ows_targ = np.tile(self.word_to_semantics[word], (iph.shape[time_dim], 1))

        return {'task' : 'comprehension_word',
                'iph' : iph,
                'ows_targ' : ows_targ}

    def production_word(self, word,
                        time_dim = 0):
        """
        """
        oph_targ = self._convert_word_to_ph_pattern(word = word)
        iws = np.tile(self.word_to_semantics[word], (oph_targ.shape[time_dim], 1))

        return {'task' : 'production_word',
                'iws' : iws,
                'oph_targ' : oph_targ}

    # -- Sentence
    def repetition(self, sentence_lst):
        """
        """
        sentence_ph = self._convert_sentence_to_ph_pattern(sentence_lst = sentence_lst)
        silence = np.zeros(sentence_ph.shape)

        return {'task' : 'repetition',
                'iph' : np.concatenate((sentence_ph, silence), axis = 0),
                'oph_targ' : np.concatenate((silence, sentence_ph), axis = 0)}

    def comprehension(self, sentence_lst,
                      time_dim = 0):
        """
        """
        iph = self._convert_sentence_to_ph_pattern(sentence_lst = sentence_lst)
        oss_targ = np.tile(self.sentence_to_sg[' '.join(sentence_lst)], (iph.shape[time_dim], 1))

        return {'task' : 'comprehension',
                'iph' : iph,
                'oss_targ' : oss_targ}

    def production(self, sentence_lst,
                   time_dim = 0):
        """
        """
        oph_targ = self._convert_sentence_to_ph_pattern(sentence_lst = sentence_lst)
        iss = np.tile(self.sentence_to_sg[' '.join(sentence_lst)], (oph_targ.shape[time_dim], 1))

        return {'task' : 'production',
                'iss' : iss,
                'oph_targ' : oph_targ}

    # Miscellaneous interface with Language-Lichtheim model
    def add_batch_dim(self, io,
                      batch_dim = 1,
                      ignore_keys = ['task']):
        """
        """
        for rep in io:
            if rep not in ignore_keys:
                io[rep] = np.expand_dims(io[rep], axis = batch_dim)

        return io

    def add_task_node(self, io,
                      time_dim = 0):
        """
        """
        io_time = io[list(io)[1]].shape[time_dim]

        if io['task'] == 'repetition':
            task_node = np.tile(np.array([1, 0, 0, 0, 0, 0]), (io_time, 1))
        elif io['task'] == 'comprehension':
            task_node = np.tile(np.array([0, 1, 0, 0, 0, 0]), (io_time, 1))
        elif io['task'] == 'production':
            task_node = np.tile(np.array([0, 0, 1, 0, 0, 0]), (io_time, 1))
        elif io['task'] == 'repetition_word':
            task_node = np.tile(np.array([0, 0, 0, 1, 0, 0]), (io_time, 1))
        elif io['task'] == 'comprehension_word':
            task_node = np.tile(np.array([0, 0, 0, 0, 1, 0]), (io_time, 1))
        elif io['task'] == 'production_word':
            task_node = np.tile(np.array([0, 0, 0, 0, 0, 1]), (io_time, 1))

        io['itn'] = task_node

        return io

    def fill_io(self, io,
                needed_reps = ['iph', 'oph_targ', 'iws', 'ows_targ', 'iss', 'oss_targ'],
                time_dim = 0):
        """
        Fill a generic task dictionary with patterns for all tasks not represented in dictionary
        """
        # Get the number of time steps over which other reps should be filled using first rep in dict
        io_time = io[list(io)[1]].shape[time_dim]

        # Fill in the remaining io with zeros
        for rep in needed_reps:
            if rep not in list(io.keys()):
                if rep == 'iph' or rep == 'oph_targ':
                    io[rep] = np.zeros((io_time, len(self.ph_to_index)))
                elif rep == 'iws' or rep == 'ows_targ':
                    io[rep] = np.zeros((io_time, len(self.ws_to_index)))
                elif rep == 'iss' or rep == 'oss_targ':
                    io[rep] = np.zeros((io_time, self.ss_io_size))

        return io

    def package_io_with_info(self, io, info,
                             requested_info = ['word', 'sentence_str', 'utterance_str', 'sentence_lst', 'utterance_lst', 'structure', 'sentence_id', 'probability', 'role_to_filler', 'filler_to_role', 'verb']):
        """
        Add information to task dictionary
        """
        info = {k : info[k] for k in requested_info if k in info}

        io.update(info)

        return io

    # Wrappers to generate multiple examples from one or more sentence lists
    def generate_weighted_tasks_for_utterance(self, utterance,
                                              info = {},
                                              task_weights = {'repetition' : 1, 'comprehension' : 3, 'production' : 2}):
        """
        Generic function to produce tasks for words or sentences
        """
        all_patterns = []
        for task in task_weights:
            # If len of the utterance is just 1 word, we are conducting word repetition, comprehension, or production
            if len(utterance) == 1:
                if task == 'repetition':
                    patterns = self.fill_io(self.repetition_word(word = utterance[0]))
                elif task == 'comprehension':
                    patterns = self.fill_io(self.comprehension_word(word = utterance[0]))
                elif task == 'production':
                    patterns = self.fill_io(self.production_word(word = utterance[0]))
            # If the length of the utterance is greater, this is sentence repetition, comprehension, or production
            else:
                if task == 'repetition':
                    patterns = self.fill_io(self.repetition(sentence_lst = utterance))
                elif task == 'comprehension':
                    patterns = self.fill_io(self.comprehension(sentence_lst = utterance))
                elif task == 'production':
                    patterns = self.fill_io(self.production(sentence_lst = utterance))

            patterns = self.add_task_node(io = patterns)

            patterns = self.add_batch_dim(io = patterns)

            patterns = self.package_io_with_info(io = patterns, info = info)

            all_patterns += [patterns] * (task_weights[task])

        return all_patterns

    def generate_weighted_tasks_for_word(self, word,
                                         info = {},
                                         task_weights = {'repetition' : 1, 'comprehension' : 3, 'production' : 2}):
        """
        """
        all_patterns = []
        for task in task_weights:
            if task == 'repetition':
                patterns = self.fill_io(self.repetition_word(word = word))
            elif task == 'comprehension':
                patterns = self.fill_io(self.comprehension_word(word = word))
            elif task == 'production':
                patterns = self.fill_io(self.production_word(word = word))

            patterns = self.add_task_node(io = patterns)

            patterns = self.add_batch_dim(io = patterns)

            patterns = self.package_io_with_info(io = patterns, info = info)

            all_patterns += [patterns] * task_weights[task]

        return all_patterns

    def generate_weighted_tasks_for_sentence(self, sentence_lst,
                                             info = {},
                                             task_weights = {'repetition' : 1, 'comprehension' : 3, 'production' : 2}):
        """
        """
        all_patterns = []
        for task in task_weights:
            if task == 'repetition':
                patterns = self.fill_io(self.repetition(sentence_lst = sentence_lst))
            elif task == 'comprehension':
                patterns = self.fill_io(self.comprehension(sentence_lst = sentence_lst))
            elif task == 'production':
                patterns = self.fill_io(self.production(sentence_lst = sentence_lst))

            patterns = self.add_task_node(io = patterns)

            patterns = self.add_batch_dim(io = patterns)

            patterns = self.package_io_with_info(io = patterns, info = info)

            all_patterns += [patterns] * task_weights[task]

        return all_patterns

    # Generate artificial word
    """
    def generate_all_artificial_words(self, num_phones,
                                      word_key = 'word'):
        phonemes = list(self.ph_to_index.keys())

        phonemes = [phonemes] * num_phones

        real_words = self.lexicon[word_key]
        artificial_words = [list(t) for t in itertools.product(*phonemes) if list(t) not in real_words]

        return artificial_words
    """

def train_test_split_probs(exs, probabilities, num_training_examples,
                           probability_key = 'probability'):
  """
  Takes set of examples and splits them into a training and testing set. Sentences are
  sampled based on their probability.
  """
  assert num_training_examples <= len(exs)

  #probabilities = [sentence.probability for sentence in exs]

  sample_indices = np.random.choice(a = range(len(exs)),
                                    size = num_training_examples,
                                    p = probabilities,
                                    replace = False)
  oos_indices = [i for i in range(len(exs)) if i not in sample_indices]

  sample = [exs[i] for i in sample_indices]
  out_of_sample = [exs[i] for i in oos_indices]

  return sample, out_of_sample


def utterances_from_file(utterances_file,
                         sep = "\t",
                         literal_evals = ['needed_roles', 'utterance_lst', 'sentence_lst', 'role_to_filler', 'filler_to_role']):
    """
    Reads utterance from a df of utterance information
    """
    utterances_df = pd.read_csv(utterances_file, sep = sep)

    # Apply literal evald
    for le in literal_evals:
        if le in utterances_df:
            utterances_df[le] = utterances_df[le].apply(ast.literal_eval)

    # Create dictionaries of sentence information
    return [row for _, row in utterances_df.iterrows()]


if __name__ == '__main__':
    language_name = 'datives'
    language_dir = f'language/{language_name}'

    interface = LMInterface(language_dir = language_dir,
                            phonology_filestem = 'phonology_artificial')

    sentences_dir = Path.cwd().joinpath(f'language/{language_name}/all_sentences.tsv')
    sentences = utterances_from_file(sentences_dir, sep = "\t")
    probabilities = [s['probability'] for s in sentences]

    intransitive_probabilities = [s['probability'] for s in sentences if s['structure'] == 'intransitive']
    print(sum(intransitive_probabilities))

    train, test = train_test_split_probs(exs = sentences, probabilities = probabilities, num_training_examples = 300)

    print(sum([s['probability'] for s in train]))
    print(sum([s['probability'] for s in test]))