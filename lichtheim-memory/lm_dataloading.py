import numpy as np

from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Corpus(Dataset):
  def __init__(self, utterances, utterance_infos, interface,
               task_weights = {'repetition' : 1, 'comprehension' : 3, 'production' : 2}):
    """
    """
    self.interface = interface

    self.utterances = utterances
    self.utterance_infos = utterance_infos

    self.task_weights = task_weights
  
  def convert_utterance_to_weighted_tasks(self, utterance, utterance_info):
    """
    """
    return self.interface.generate_weighted_tasks_for_utterance(utterance = utterance, 
                                                                info = utterance_info,
                                                                task_weights = self.task_weights)

  def __len__(self):
    """
    """
    return len(self.utterances)

  def __getitem__(self, idx):
    """
    """
    return self.convert_utterance_to_weighted_tasks(utterance = self.utterances[idx],
                                                    utterance_info = self.utterance_infos[idx])

def train_test_split_rand(exs, num_training_examples):
  """
  Samples examples into training and testing sets without respect to probabilities of examples
  """
  assert num_training_examples <= len(exs)

  sample_indices = np.random.choice(a = range(len(exs)),
                                    size = num_training_examples,
                                    replace = False)
  oos_indices = [i for i in range(len(exs)) if i not in sample_indices]

  sample = [exs[i] for i in sample_indices]
  out_of_sample = [exs[i] for i in oos_indices]

  return sample, out_of_sample

def train_test_split_probs(exs, probabilities, num_training_examples,
                           probability_key = 'probability'):
  """
  Takes set of examples and splits them into a training and testing set. Sentences are
  sampled based on their probability.
  """
  assert num_training_examples <= len(exs)

  # Normalize probabilities
  prob_sum = sum(probabilities)
  probabilities = [p / prob_sum for p in probabilities]

  sample_indices = np.random.choice(a = range(len(exs)),
                                    size = num_training_examples,
                                    p = probabilities,
                                    replace = False)
  oos_indices = [i for i in range(len(exs)) if i not in sample_indices]

  sample = [exs[i] for i in sample_indices]
  out_of_sample = [exs[i] for i in oos_indices]

  return sample, out_of_sample

def train_test_split_probs_binned(utterances, tt_split,
                                  probability_key = 'probability',
                                  bin_tt_label = 'structure'):
  """
  """
  utterances_training, utterances_testing = [], []

  for bin_category in set([utterance[bin_tt_label] for utterance in utterances]):
    binned_utterances = [utterance for utterance in utterances if utterance[bin_tt_label] == bin_category]
    bin_category_training, bin_category_testing = train_test_split_probs(exs = binned_utterances,
                                                                         probabilities = [binned_utterances['probability'] for binned_utterances in binned_utterances],
                                                                         num_training_examples = int(len(binned_utterances) * tt_split))
    utterances_training += bin_category_training
    utterances_testing += bin_category_testing

  return utterances_training, utterances_testing


def inst_training_env(words, utterances, tt_split, interface,
                      sample_tt_probability = True,
                      bin_tt_label = 'structure',
                      probability_key = 'probability',
                      seed = 9):
  """
  """
  np.random.seed(seed)

  # Generate examples and dataloader
  #-- Words
  corpus_words = Corpus(utterances = [[word] for word in words], 
                        utterance_infos = [{'word' : word} for word in words], 
                        interface = interface)

  dataloader_words = DataLoader(corpus_words,
                                batch_size = None)

  #-- Sentences
  # -- # -- Split training and testing sentences
  if sample_tt_probability:
    if bin_tt_label: # Here we split the set based on a bin category
      utterances_training, utterances_testing = train_test_split_probs_binned(utterances = utterances, tt_split = tt_split,
                                                                              probability_key = probability_key,
                                                                              bin_tt_label = bin_tt_label)


    else: # Otherwise we split without taking into account a bin
      utterances_training, utterances_testing = train_test_split_probs(exs = utterances,
                                                                       probabilities = [utterance['probability'] for utterance in utterances],
                                                                       num_training_examples = int(len(utterances) * tt_split))
  else:
    utterances_training, utterances_testing = train_test_split_rand(exs = utterances,
                                                                    num_training_examples = int(len(utterances) * tt_split))

  # -- # -- Training set definition
  corpus_sentences_training = Corpus(utterances = [sentence['sentence_lst'] for sentence in utterances_training], 
                                     utterance_infos = utterances_training, 
                                     interface = interface)

  sampler = WeightedRandomSampler(weights = [sentence['probability'] for sentence in utterances_training], 
                                  num_samples = len(corpus_sentences_training), 
                                  replacement = True)
  dataloader_sentences_training = DataLoader(corpus_sentences_training,
                                             batch_size = None,
                                             sampler = sampler)

  # -- # -- Testing set definition
  corpus_sentences_testing = Corpus(utterances = [sentence['sentence_lst'] for sentence in utterances_testing], 
                                    utterance_infos = utterances_testing, 
                                    interface = interface)
  dataloader_sentences_testing = DataLoader(corpus_sentences_testing,
                                            batch_size = None)
  
  return dataloader_words, dataloader_sentences_training, dataloader_sentences_testing, utterances_training, utterances_testing