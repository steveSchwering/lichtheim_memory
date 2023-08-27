import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from torch import save

# General functions
def mean(lst):
    return sum(lst) / len(lst)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Model information logging
def log_model_init(model, language_dir, seed, sent_train, sent_test,
                   log_path = 'model_info',
                   log_file = 'lm_model_logger.csv',
                   corpus_file = 'corpus_info.csv'):
  """
  """
  model_logger_path = Path.cwd().joinpath(log_path).joinpath(log_file)

  # Get information about model weights and language
  model_info = {'datetime' : datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 'model_id' : model.model_id, 'seed' : seed, 'language' : language_dir}
  model_info.update(model.get_model_weight_sizes())

  # Save model information
  if model_logger_path.is_file():
    pd.DataFrame([model_info]).to_csv(model_logger_path, mode = 'a', header = False, index = False)
  else:
    pd.DataFrame([model_info]).to_csv(model_logger_path, header = True, index = False)

  # Save info about training and testing sentences
  corpus_path = Path.cwd().joinpath(log_path).joinpath(model.model_id)
  corpus_path.mkdir(parents = True, exist_ok = True)

  train_df = pd.DataFrame([sent for sent in sent_train])
  test_df = pd.DataFrame([sent for sent in sent_test])

  train_df.to_csv(corpus_path.joinpath(f'train_{corpus_file}'), header = True, index = False)
  test_df.to_csv(corpus_path.joinpath(f'test_{corpus_file}'), header = True, index = False)

def log_model_state(model, trained_epochs, training_meta_info,
                    log_path = 'model_info',
                    trained_epochs_key = 'trained_epochs'):
  """
  """
  model_path = model_object_path = Path.cwd().joinpath(log_path).joinpath(model.model_id)

  # Model itself is saved in a folder
  model_object_path = model_path.joinpath(f'models/{model.model_id}_e{trained_epochs}.pt')
  model_object_path.parent.mkdir(parents = True, exist_ok = True)
  save(model.state_dict(), model_object_path)

  # Information about model is saved in log for that model
  model_logger_path = model_path.joinpath('model_log.csv')
  model_logger_path.parent.mkdir(parents = True, exist_ok = True)

  model_info = {'datetime' : datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 'model_id' : model.model_id, 'trained_epochs' : trained_epochs}
  model_info.update(model.get_model_weight_sizes())
  model_info.update(training_meta_info)
  model_info.update({'model_object_path' : model_object_path})

  if model_logger_path.is_file():
    pd.DataFrame([model_info]).to_csv(model_logger_path, mode = 'a', header = False, index = False)
  else:
    pd.DataFrame([model_info]).to_csv(model_logger_path, header = True, index = False)


def log_model_behavior(model, behavior, meta_info, trained_epochs, behavior_type,
                       log_path = 'model_info'):
  """
  """
  model_path = Path.cwd().joinpath(log_path).joinpath(model.model_id)
  
  # Append meta information
  model_info = {'datetime' : datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 'model' : model.model_id, 'behavior_type' : behavior_type}
  model_info.update(meta_info)
  for b in behavior:
    b.update(model_info)

  # Saving behavior to disk
  model_behavior_path = model_path.joinpath(f'behavior/{behavior_type}/{model.model_id}_e{trained_epochs}_behavior_{behavior_type}.csv')
  model_behavior_path.parent.mkdir(parents = True, exist_ok = True)
  behav_df = pd.DataFrame(behavior)
  behav_df.to_csv(model_behavior_path, index = False)


def log_model_loss(model, loss, meta_info, trained_epochs, behavior_type,
                   log_path = 'model_info'):
  """
  """
  model_path = Path.cwd().joinpath(log_path).joinpath(model.model_id)

  # Append meta information
  model_info = {'datetime' : datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 'model' : model.model_id, 'behavior_type' : behavior_type}
  model_info.update(meta_info)
  for l in loss:
    l.update(model_info)

  # Saving losses to disk
  model_loss_path = model_path.joinpath(f'loss/{behavior_type}/lm_{model.model_id}_e{trained_epochs}_loss_{behavior_type}.csv')
  model_loss_path.parent.mkdir(parents = True, exist_ok = True)
  loss_df = pd.DataFrame(loss)
  loss_df.to_csv(model_loss_path, index = False)

# Model behavior logging: Functions to work with pytorch
def detach_and_numpize_tensor(t):
  """
  Turns a pytorch tenor into a numpy array
  """
  if t.requires_grad:
    return t.detach().numpy()
  else:
    return t.numpy()

def detach_and_numpize_tensors(behavior):
  """
  Takes an example and converts the tensors in the example into numpy arrays

  behavior is expected to contain a number of pytorch tensors
  """
  for key in behavior:
    if torch.is_tensor(behavior[key]):
      behavior[key] = detach_and_numpize_tensor(t = behavior[key])
  
  return behavior

# Package output into dataframe
def split_key_activations(activations, prefix,
                          between_sep = '_',
                          within_sep = '.'):
  """
  Splits keys and values into separate columns
  e.g. oph_max.0_id
  """
  packaged_key_activations = {}

  for n, (k, v) in enumerate(activations.items()):
    packaged_key_activations[f'{prefix}{between_sep}max{within_sep}{n}{between_sep}id'] = k # Getting the ID
    packaged_key_activations[f'{prefix}{between_sep}max{within_sep}{n}{between_sep}ac'] = v # Getting the activation
  
  return packaged_key_activations

def prefix_keys(d, prefix, 
                within_sep = '.'):
  """
  Takes a dictionary and adds a prefix to all keys
  Prefix is frequently the name of the datatype, and key is the name of the value
  e.g. oph.C1-0
  """
  nd = {}
  for k, v in d.items():
    nd[f'{prefix}{within_sep}{k}'] = v
  
  return nd

def n_highest_activations(activations, n):
  """
  Returns n highests elements of dictionary
  """
  n_highest_keys = sorted(activations, key = activations.get, reverse = True)[:n]

  return dict((k, activations[k]) for k in n_highest_keys)

def interpret_all_rep_activations(rep, interface, activations, batch, timestep):
  """
  Returns dictionary of representation : activation pairs

  + rep: string of ph, ws, or ss denoting type of representation being assigned
  + interface: object containing 
  """
  # Get list of representations
  if rep == 'ph':
    reps = list(interface.ph_to_index.keys())
  elif rep == 'ws':
    reps = list(interface.ws_to_index.keys())
  elif rep == 'ss':
    reps = list(range(interface.ss_io_size))
  
  # Get activations
  activations = activations[timestep][batch].tolist()

  # Assign activations to representation, index-wise
  rep_to_ac = {}
  for rep, ac in zip(reps, activations):
    rep_to_ac[rep] = f"{ac:.5f}"

  return rep_to_ac

def package_ph(behavior, interface, batch, timestep,
               n_highest = 1,
               prefix_iph = 'iph',
               prefix_oph = 'oph',
               prefix_toph = 'toph'):
  """
  """
  iph = behavior['iph']
  oph = behavior['oph']
  oph_targ = behavior['oph_targ']

  # Put all activations into a dictionary and format
  activations_iph = interpret_all_rep_activations(rep = 'ph', interface = interface, activations = iph, batch = 0, timestep = timestep)
  activations_oph = interpret_all_rep_activations(rep = 'ph', interface = interface, activations = oph, batch = 0, timestep = timestep)
  activations_oph_targ = interpret_all_rep_activations(rep = 'ph', interface = interface, activations = oph_targ, batch = 0, timestep = timestep)

  # Put n highest activations into dictionary
  highest_oph = n_highest_activations(activations_oph, n = n_highest)
  highest_oph_targ = n_highest_activations(activations_oph_targ, n = n_highest)

  highest_oph_info = split_key_activations(activations = highest_oph, prefix = prefix_oph)
  highest_oph_targ_info = split_key_activations(activations = highest_oph_targ, prefix = prefix_toph)

  activations_iph = prefix_keys(activations_iph, prefix = prefix_iph)
  activations_oph = prefix_keys(activations_oph, prefix = prefix_oph)
  activations_oph_targ = prefix_keys(activations_oph_targ, prefix = prefix_toph)

  return {**activations_iph, **activations_oph, **activations_oph_targ, **highest_oph_info, **highest_oph_targ_info}

def package_ws(behavior, interface, batch, timestep,
               n_highest = 1,
               prefix_iws = 'iws',
               prefix_ows = 'ows',
               prefix_tows = 'tows'):
  """
  """
  iws = behavior['iws']
  ows = behavior['ows']
  ows_targ = behavior['ows_targ']

  # Put all activations into dictionary and format
  activations_iws = interpret_all_rep_activations(rep = 'ws', interface = interface, activations = iws, batch = 0, timestep = timestep)
  activations_ows = interpret_all_rep_activations(rep = 'ws', interface = interface, activations = ows, batch = 0, timestep = timestep)
  activations_ows_targ = interpret_all_rep_activations(rep = 'ws', interface = interface, activations = ows_targ, batch = 0, timestep = timestep)
  
  activations_iws = prefix_keys(activations_iws, prefix = prefix_iws)
  activations_ows = prefix_keys(activations_ows, prefix = prefix_ows)
  activations_ows_targ = prefix_keys(activations_ows_targ, prefix = prefix_tows)

  # Put n highest activations into dictionary and format
  highest_ows = n_highest_activations(activations_ows, n = n_highest)
  highest_ows_targ = n_highest_activations(activations_ows_targ, n = n_highest)

  highest_ows_info = split_key_activations(activations = highest_ows, prefix = prefix_ows)
  highest_ows_targ_info = split_key_activations(activations = highest_ows_targ, prefix = prefix_tows)

  return {**activations_iws, **activations_ows, **activations_ows_targ, **highest_ows_info, **highest_ows_targ_info}

def package_ss(behavior, interface, batch, timestep,
               prefix_iss = 'iss',
               prefix_oss = 'oss',
               prefix_toss = 'toss'):
  """
  """
  iss = behavior['iss']
  oss = behavior['oss']
  oss_targ = behavior['oss_targ']

  # Put all activations into dictionary format
  activations_iss = interpret_all_rep_activations(rep = 'ss', interface = interface, activations = iss, batch = 0, timestep = timestep)
  activations_oss = interpret_all_rep_activations(rep = 'ss', interface = interface, activations = oss, batch = 0, timestep = timestep)
  activations_oss_targ = interpret_all_rep_activations(rep = 'ss', interface = interface, activations = oss_targ, batch = 0, timestep = timestep)

  activations_iss = prefix_keys(activations_iss, prefix = prefix_iss)
  activations_oss = prefix_keys(activations_oss, prefix = prefix_oss)
  activations_oss_targ = prefix_keys(activations_oss_targ, prefix = prefix_toss)

  return {**activations_iss, **activations_oss, **activations_oss_targ}

def package_output(behavior, interface,
                   n_highest = 5,
                   time_dim = 0):
  """
  """
  timesteps = behavior['iph'].shape[time_dim]

  all_activations = []
  for t in range(timesteps):
    activations = {}

    # Track base word information
    if 'sentence_str' in behavior: activations['sentence'] = behavior['sentence_str']
    if 'utterance_str' in behavior: activations['utterance'] = behavior['utterance_str']
    if 'word' in behavior: activations['word'] = behavior['word']
    if 'task' in behavior: activations['task'] = behavior['task']
    if 'structure' in behavior: activations['structure'] = behavior['structure']
    if 'probability' in behavior: activations['probability'] = behavior['probability']
    activations['t'] = t
    activations['max_length'] = timesteps

    # Phonology input, output, and target output
    ph_behavior = package_ph(behavior = behavior, interface = interface, batch = 0, timestep = t, n_highest = n_highest)
    activations.update(ph_behavior)

    # Word semantic input, output, and target output
    ws_behavior = package_ws(behavior = behavior, interface = interface, batch = 0, timestep = t, n_highest = n_highest)
    activations.update(ws_behavior)

    # Sentence semantic input, output, and target behavior
    ss_behavior = package_ss(behavior = behavior, interface = interface, batch = 0, timestep = t)
    activations.update(ss_behavior)

    all_activations.append(activations)
  
  return all_activations

def package_all_output(behavior, interface,
                       n_highest = 5):
  """
  """
  all_activations = []

  for b in behavior:
    all_activations += package_output(behavior = b, interface = interface,
                                      n_highest = n_highest)
  
  return all_activations