import copy

import torch.nn as nn

import lm_logging

from torch import stack

# Stringify loss for report
def str_ex(utterance_str, task,
           task_start = 0,
           task_chars = 3):
  """
  Returns string formatted utterance information
  """
  return f'{utterance_str} ({task[task_start:task_start + task_chars]})'

def str_loss(loss_ph, loss_ws, loss_ss):
  """
  Returns string formatted loss information
  """
  return f'PH {loss_ph:.5f} | WS {loss_ws:.5f} | SS {loss_ss:.5f}'

def report_ex_loss(behavior):
  """
  Prints string formatted utterance and loss information
  """
  # Get example information
  try:
    ex_info = str_ex(utterance_str = behavior['word'], task = behavior['task'])
  except KeyError:
    ex_info = str_ex(utterance_str = behavior['sentence_str'], task = behavior['task'])
  
  # Get loss information
  loss_info = str_loss(loss_ph = behavior['loss_ph'], loss_ws = behavior['loss_ws'], loss_ss = behavior['loss_ss'])
  
  print(f"{ex_info} - {loss_info}")

def summarize_exs_loss(behavior,
                       reported_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word']):
  """
  Prints string formatted utterance and loss information for each task
  """
  for task in reported_tasks:
    loss_ph = []
    loss_ws = []
    loss_ss = []

    for ex in behavior:
      if task != ex['task']: continue
      loss_ph += [ex['loss_ph'].item()]
      loss_ws += [ex['loss_ws'].item()]
      loss_ss += [ex['loss_ss'].item()]

    try:
      print(f"\t{task.title()} -- Loss ph: {sum(loss_ph) / len(loss_ph)}")
      print(f"\t{task.title()} -- Loss ws: {sum(loss_ws) / len(loss_ws)}")
      print(f"\t{task.title()} -- Loss ss: {sum(loss_ss) / len(loss_ss)}")
    except ZeroDivisionError:
      print(f'\t{task.title()} -- NA')

# Extract loss -- for tracking
def report_loss_from_example(behavior,
                             loss_types = ['loss_ph', 'loss_ws', 'loss_ss'],
                             ex_info_tracked = ['sentence_str', 'word', 'utterance', 'task', 'structure', 'probability']):
  """
  Returns dictionary tracking critical loss information from example behavior
  """
  all_loss_info_in_example = []
  ex_info = {ei : behavior[ei] for ei in ex_info_tracked if ei in behavior.keys()}

  for loss_type in loss_types:
    loss_info = {'loss_type' : loss_type, 'loss' : {behavior[loss_type].item()}}
    loss_info.update(ex_info)
    all_loss_info_in_example.append(loss_info)

  return all_loss_info_in_example

def report_loss_from_examples(examples,
                              info_to_append = {'epoch' : None}):
  """
  Returns losses where each row tracks for group of examples
  """
  all_losses = []

  for ex_num, behavior in enumerate(examples):
    loss_info = report_loss_from_example(behavior = behavior)

    for lf in loss_info:
      lf['ex_num'] = ex_num
      lf.update(info_to_append)
  
    all_losses += loss_info

  return all_losses

# Extract loss -- for training
def organize_loss_factors(behavior):
  """
  Gets outputs for all timesteps and ensures they are same shape as all target outputs
  """
  # Get model outputs -- stack outputs through time
  oph = behavior['oph']
  ows = behavior['ows']
  oss = behavior['oss']

  # Get targets -- already stacked through time
  oph_targ = behavior['oph_targ']
  ows_targ = behavior['ows_targ']
  oss_targ = behavior['oss_targ']

  assert oph.shape == oph_targ.shape
  assert ows.shape == ows_targ.shape
  assert oss.shape == oss_targ.shape

  return oph, ows, oss, oph_targ, ows_targ, oss_targ

def calculate_loss(behavior,
                   ph_criterion = nn.BCELoss(),
                   ws_criterion = nn.BCELoss(),
                   ss_criterion = nn.MSELoss()):
  """
  """
  # Gather loss factors
  oph, ows, oss, oph_targ, ows_targ, oss_targ = organize_loss_factors(behavior = behavior)

  # Output phonology loss
  loss_ph = ph_criterion(oph.float(), oph_targ.float())

  # Output word semantics loss
  loss_ws = ws_criterion(ows.float(), ows_targ.float())

  # Output sentence semantics loss
  loss_ss = ss_criterion(oss.float(), oss_targ.float())

  return {'loss_ph' : loss_ph,
          'loss_ws' : loss_ws,
          'loss_ss' : loss_ss}

def backpropagate(losses, optimizer, task = None):
  """
  SHOULD LOSS BE SELECTIVELY BACKPROPAGATED DEPENDING ON TASK?

  e.g. in Ueno et al.: "In repetition, this layer [outputting semantics during 
  comprehension] was not assigned a specific role and so its activations were 
  unconstrained."

  During a sentence comprehension task, the model is not penalized on word semantics
  """
  # Combine loss depending on task
  if task: # Train on specific loss based on task
    if 'repetition' in task:
      train_loss = losses['loss_ph']
    elif 'production' in task:
      train_loss = losses['loss_ph']
    elif 'comprehension' in task:
      if 'word' in task: # Word comprehension
        train_loss = losses['loss_ws']
      else:              # Sentence comprehension
        train_loss = losses['loss_ss']
  else:
    train_loss = sum(losses.values())

  # Update model
  train_loss.backward()
  optimizer.step()

  return losses

def stack_activations(activations):
  """
  Takes a list of dictionaries containing key:vector pairs, stacking vectors by key
  """
  keys = list(activations[0].keys())

  stacked_activations = {}
  for key in keys:
    stacked_activations[key] = stack([a[key] for a in activations])

  return stacked_activations

def forward_timesteps(model, example,
                      prev_o = None,
                      needed_prev_o = ['oph', 'ows', 'oss', 'phh1', 'wsh1', 'wsh2', 'ssh1', 'ssh2', 'ssh3']):
  """
  Runs through all timesteps for a single task for the model
  """
  if not prev_o:
    prev_o = {key: None for key in needed_prev_o}

  all_timestep_activations = []

  # Iterate through timesteps
  for t_iph, t_iws, t_iss, t_itn in zip(example['iph'], example['iws'], example['iss'], example['itn']):

    # Run model through one timestep
    timestep_activations = model(iph = t_iph, iws = t_iws, iss = t_iss, itn = t_itn,
                                 task = example['task'],
                                 prev_oph = prev_o['oph'], prev_ows = prev_o['ows'], prev_oss = prev_o['oss'],
                                 prev_phh1 = prev_o['phh1'],
                                 prev_wsh1 = prev_o['wsh1'], prev_wsh2 = prev_o['wsh2'],
                                 prev_ssh1 = prev_o['ssh1'], prev_ssh2 = prev_o['ssh2'], prev_ssh3 = prev_o['ssh3'])
    prev_o = timestep_activations
    all_timestep_activations.append(timestep_activations)
  
  # Return activations of model
  return stack_activations(all_timestep_activations)

def run_example(model, example,
                train_flag = False,
                optimizer = None,
                report_flag = False):
  """
  Runs an example through model, returning all example information along with output behavior
  """
  # Create a copy of the example dictionary
  behavior = copy.deepcopy(example)

  # Reset gradient and run example
  if train_flag: optimizer.zero_grad()
  activations = forward_timesteps(model = model, example = behavior, prev_o = None)
  behavior.update(activations)

  # Calculate loss and train
  losses = calculate_loss(behavior = behavior)
  if train_flag: backpropagate(losses = losses, optimizer = optimizer, task = behavior['task'])
  behavior.update(losses)

  # Report
  if report_flag: report_ex_loss(behavior = behavior)

  return behavior

def run_examples(model, examples,
                 filter_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word'],
                 train_flag = False,
                 optimizer = None,
                 report_flag = False):
  """
  Run a list of examples through model
  """
  all_behavior = []

  for example in examples:

    # Skip example if task not in filter
    if example['task'] not in filter_tasks: continue

    # Run example
    behavior = run_example(model = model, example = example,
                           train_flag = train_flag,
                           optimizer = optimizer,
                           report_flag = report_flag)
    all_behavior.append(behavior)
    
  return all_behavior

def run_epoch(model, dataloader,
              optimizer = None,
              scheduler = None,
              train_flag = False,
              filter_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word'],
              report_flag = False):
  """
  Run through multiple sets of examples
  """
  all_behaviors = []
  # Run through all examples in dataloader
  for examples in dataloader:
    all_example_behavior = run_examples(model = model, examples = examples,
                                        filter_tasks = filter_tasks,
                                        train_flag = train_flag,
                                        optimizer = optimizer,
                                        report_flag = report_flag)
    all_behaviors += all_example_behavior

  # If scheduler is present, advance it after epoch
  if scheduler:
    scheduler.step()
  
  return all_behaviors

def train_epoch(model, dataloader, optimizer,
                scheduler = None,
                filter_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word'],
                report_flag = False):
  """
  Trains model for 1 epoch
  """
  return run_epoch(model = model, dataloader = dataloader, optimizer = optimizer,
                   scheduler = scheduler,
                   train_flag = True,
                   filter_tasks = filter_tasks,
                   report_flag = report_flag)

def test_epoch(model, dataloader,
               epoch_num = None,
               filter_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word'],
               report_flag = False):
  """
  """
  all_behaviors = run_epoch(model = model, dataloader = dataloader,
                            optimizer = None,
                            scheduler = None,
                            train_flag = False,
                            filter_tasks = filter_tasks,
                            report_flag = report_flag)

  loss_tracker = report_loss_from_examples(examples = all_behaviors, info_to_append = {'trained_epochs' : epoch_num})

  return all_behaviors, loss_tracker

def train_n_epochs(model, dataloader, optimizer, n,
                   scheduler = None,
                   filter_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word'],
                   report_flag = False,
                   epoch_key = 'epoch'):
  """
  """
  behaviors = []
  loss_tracker = []

  for epoch in range(n):
    b = train_epoch(model = model, dataloader = dataloader, optimizer = optimizer,
                    scheduler = scheduler,
                    filter_tasks = filter_tasks,
                    report_flag = report_flag)
    behaviors += b
    loss_tracker += report_loss_from_examples(examples = b, info_to_append = {epoch_key : epoch})

  return behaviors, loss_tracker

def interleave_training(model, dataloader_sentences, dataloader_words, num_epochs, optimizer, scheduler,
                        word_epochs_per_sentence_epoch = 3,
                        word_epochs_decay_stepsize = 1,
                        word_epochs_decay_steprate = 10,
                        filter_tasks = ['repetition', 'comprehension', 'production', 'repetition_word', 'comprehension_word', 'production_word'],
                        report_every = 10,
                        summarize_every = 10,
                        log_every = 100000,
                        training_meta_info = {'seed' : None, 'language_dir' : None}):
  """
  Trains model on both sentences and words
  """
  all_loss_word = []
  all_loss_sentence = []

  for epoch in range(num_epochs):
    
    # Determine if report will happen
    if epoch and epoch % report_every == 0:
      report_flag = True
    else:
      report_flag = False

    # Decay number of word epochs to prepend
    if (epoch != 0) & (epoch % word_epochs_decay_steprate == 0) & (word_epochs_per_sentence_epoch > 0):
        word_epochs_per_sentence_epoch -= word_epochs_decay_stepsize
        training_meta_info['word_epochs_per_sentence_epoch'] = word_epochs_per_sentence_epoch

    # Traing model on all words word_epoch times -- scheduler is None because we do not want to advance it during these epochs
    _, loss_word = train_n_epochs(model = model, dataloader = dataloader_words, optimizer = optimizer, n = word_epochs_per_sentence_epoch,
                                  scheduler = None,
                                  filter_tasks = filter_tasks,
                                  report_flag = report_flag,
                                  epoch_key = 'epoch_sub_word')
    loss_word = [loss_word_info.update({'epoch' : epoch}) for loss_word_info in loss_word] # Add loss information
    all_loss_word += loss_word
    
    # Train model on sentences -- scheduler is passed because we want to update lr through these trials
    behaviors_sentence = train_epoch(model = model, dataloader = dataloader_sentences, optimizer = optimizer,
                                     scheduler = scheduler,
                                     filter_tasks = filter_tasks,
                                     report_flag = report_flag)
    all_loss_sentence += report_loss_from_examples(examples = behaviors_sentence, info_to_append = {'epoch' : epoch})
    training_meta_info['lr'] = optimizer.param_groups[0]['lr']

    if epoch and epoch % summarize_every == 0:
      print(f"Epoch {epoch}")
      summarize_exs_loss(behaviors_sentence)

    if epoch % log_every == 0:
      lm_logging.log_model_state(model = model,
                                 trained_epochs = epoch,
                                 training_meta_info = training_meta_info)
  
  return all_loss_word, all_loss_sentence, training_meta_info