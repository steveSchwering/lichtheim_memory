import random, uuid
import torch.nn as nn

from torch import zeros, cat, full, Tensor

from torch import manual_seed
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD

class LichtheimMemory(nn.Module):
  def __init__(self, ph_size, ws_size, ss_size,
               phh1_size,
               wsh1_size, wsh2_size,
               ssh1_size, ssh2_size, ssh3_size,
               task_size,
               model_id = None,
               feature_dim = 1):
    """
    Instantiates the weights of the Lichtheim model
    """
    super(LichtheimMemory, self).__init__()

    # Model ID
    self.model_id = model_id

    # Size of layers
    self.ph_size, self.ws_size, self.ss_size = ph_size, ws_size, ss_size
    # -- Dorsal stream
    self.phh1_size = phh1_size
    # -- Ventral stream
    self.wsh1_size, self.wsh2_size = wsh1_size, wsh2_size
    self.ssh1_size, self.ssh2_size, self.ssh3_size = ssh1_size, ssh2_size, ssh3_size
    # -- Meta
    self.task_size = task_size

    # Weights
    # -- Dorsal stream
    self.to_phh1 = nn.Linear(ph_size + ph_size + phh1_size, phh1_size) # Input from phonological input, feedback from output phonology, and recurrence
    # -- Ventral stream
    self.to_wsh1 = nn.Linear(ph_size + wsh2_size, wsh1_size) # Input from phonological input and feedback from wsh2
    self.to_wsh2 = nn.Linear(wsh1_size + ws_size + ssh1_size, wsh2_size) # Input from wsh1, word semantics input, and feedback from wsh3
    self.to_ows = nn.Linear(wsh2_size + task_size, ws_size) # Input from wsh2
    self.to_ssh1 = nn.Linear(wsh2_size + ssh2_size, ssh1_size) # Input from wsh2 and feedback from ssh2
    self.to_ssh2 = nn.Linear(ssh1_size + ss_size + ssh3_size, ssh2_size) # Input from ssh1, sentence semantics input, and feedback from ssh3
    self.to_oss = nn.Linear(ssh2_size + task_size, ss_size) # Input from ssh2
    self.to_ssh3 = nn.Linear(ssh2_size + ph_size, ssh3_size) # Input from ssh2 and feedback from output phonology
    # -- Combining at phonology output
    self.to_oph = nn.Linear(ssh3_size + phh1_size + task_size, ph_size) # Input from phh1 and wsh3

    # Activation functions
    self.activ_phh1 = nn.Sigmoid()
    self.activ_wsh1 = nn.Sigmoid()
    self.activ_wsh2 = nn.Sigmoid()
    self.activ_ows = nn.Sigmoid()
    self.activ_ssh1 = nn.Sigmoid()
    self.activ_ssh2 = nn.Sigmoid()
    self.activ_oss = nn.Sigmoid()
    self.activ_ssh3 = nn.Sigmoid()
    self.activ_oph = nn.Softmax(dim = feature_dim)
    #self.activ_oph = nn.Sigmoid()

  def forward(self, iph, iws, iss, itn,
              prev_oph, prev_ows, prev_oss, prev_phh1, prev_wsh1, prev_wsh2, prev_ssh1, prev_ssh2, prev_ssh3,
              task,
              batch_size = 1,
              feature_dim = 1,
              ignore_itn = True):
    """
    One forward pass through the model for 1 time step
    """
    # Decide whether to zero task nodes
    if ignore_itn:
      itn = zeros(itn.shape)

    # Generate a default value for time 0 input for recurrent connections
    if not isinstance(prev_phh1, Tensor):
      prev_oph, prev_ows, prev_oss, prev_phh1, prev_wsh1, prev_wsh2, prev_ssh1, prev_ssh2, prev_ssh3 = self._get_t0_prev_inputs(batch_size = batch_size)

    # Assign input semantics depending on task
    iws, iss = self.mask_semantics(iws, iss, prev_ows, prev_oss, task)

    # Generate activities
    # -- Repetition
    phh1_input = cat((iph, prev_oph, prev_phh1), 
                           feature_dim) # phh1 is the site of our self-recurrent connection
    phh1 = self.activ_phh1(self.to_phh1(phh1_input))

    # -- Comprehension word
    wsh1_input = cat((iph, prev_wsh2), 
                           feature_dim)
    wsh1 = self.activ_wsh1(self.to_wsh1(wsh1_input))
    wsh2_input = cat((wsh1, iws, prev_ssh1), 
                           feature_dim)
    wsh2 = self.activ_wsh2(self.to_wsh2(wsh2_input))
    ows_input = cat((wsh2, itn),
                          feature_dim)
    ows = self.activ_ows(self.to_ows(ows_input))

    # -- Comprehension sentence
    ssh1_input = cat((wsh2, prev_ssh2),
                           feature_dim)
    ssh1 = self.activ_ssh1(self.to_ssh1(ssh1_input))
    ssh2_input = cat((ssh1, iss, prev_ssh3),
                           feature_dim)
    ssh2 = self.activ_ssh2(self.to_ssh2(ssh2_input))
    oss_input = cat((ssh2, itn),
                          feature_dim)
    oss = self.activ_oss(self.to_oss(oss_input))

    # -- Production
    ssh3_input = cat((ssh2, prev_oph),
                           feature_dim)
    ssh3 = self.activ_ssh3(self.to_ssh3(ssh3_input))
    oph_input = cat((ssh3, phh1, itn),
                           feature_dim)
    oph = self.activ_oph(self.to_oph(oph_input))

    activations = {'iph' : iph, 'iws' : iws, 'iss' : iss, 'itn' : itn,
                   'oph' : oph, 'ows' : ows, 'oss' : oss,
                   'phh1' : phh1,
                   'wsh1' : wsh1, 'wsh2' : wsh2,
                   'ssh1' : ssh1, 'ssh2' : ssh2, 'ssh3' : ssh3}

    return activations

  def weights_init(self):
    """
    Defines weight initialization in the nn.Module for Linear layers: see https://blog.paperspace.com/pytorch-101-advanced/
    """
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a = -1, b = 1)
        nn.init.constant_(module.bias, val = -1)

  def _get_t0_prev_inputs(self,
                          batch_size = 1,
                          default_oph = 0.0,
                          default_other = 0.5):
    """
    Returns arrays of 0s for previous input from timestep -1
    """
    prev_oph = full((batch_size, self.ph_size), default_oph)
    prev_ows = full((batch_size, self.ws_size), default_other)
    prev_oss = full((batch_size, self.ss_size), default_other)

    prev_phh1 = full((batch_size, self.phh1_size), default_other)

    prev_wsh1 = full((batch_size, self.wsh1_size), default_other)
    prev_wsh2 = full((batch_size, self.wsh2_size), default_other)

    prev_ssh1 = full((batch_size, self.ssh1_size), default_other)
    prev_ssh2 = full((batch_size, self.ssh2_size), default_other)
    prev_ssh3 = full((batch_size, self.ssh3_size), default_other)

    return prev_oph, prev_ows, prev_oss, prev_phh1, prev_wsh1, prev_wsh2, prev_ssh1, prev_ssh2, prev_ssh3
  
  def mask_semantics(self, iws, iss, prev_ows, prev_oss, task):
    """
    If task is production, return input semantics
    + If sentence production, only sentence semantics input is treated as veridical
    + If word production, only word semantics input is treated as veridical

    If task is not production, return previous semantics

    Returned value assigned to input word semantics

    This ensures that, during production tasks, the model is receiving veridical input on its objective
    and during other tasks the model can use the output semantics as input for the next time step
    """
    if 'production' in task:
      if 'word' in task:
        return iws, prev_oss
      else:
        return prev_ows, iss
    else:
      return prev_ows, prev_oss

  def get_model_weight_sizes(self):
    """
    """
    model_weights = {'ph_size' : self.ph_size, 'ws_size' : self.ws_size, 'ss_size' : self.ss_size,
                     'phh1_size' : self.phh1_size, 
                     'wsh1_size' : self.wsh1_size, 'wsh2_size' : self.wsh2_size,
                     'ssh1_size' : self.ssh1_size, 'ssh2_size' : self.ssh2_size, 'ssh3_size' : self.ssh3_size}

    return model_weights

def inst_LM(interface,
            phh1_size, wsh1_size, wsh2_size,
            ssh1_size, ssh2_size, ssh3_size,
            lr, weight_decay, lr_steprate, lr_gamma,
            seed = 9,
            model_id = None):
  """
  """
  manual_seed(seed)
  rd = random.Random()
  rd.seed(seed)

  # Defining model
  if not model_id:
    model_id = f'lm_{seed}_{datetime.now().strftime("%Y-%m-%d")}_{uuid.UUID(int = rd.getrandbits(128), version  = 4)}'
  ph_size = len(interface.index_to_ph)
  ws_size = len(interface.index_to_ws)
  ss_size = interface.ss_io_size
  LM = LichtheimMemory(ph_size = ph_size, ws_size = ws_size, ss_size = ss_size,
                       phh1_size = phh1_size,
                       wsh1_size = wsh1_size, wsh2_size = wsh2_size,
                       ssh1_size = ssh1_size, ssh2_size = ssh2_size, ssh3_size = ssh3_size,
                       model_id = model_id,
                       task_size = 6)
  LM.weights_init()

  # Set the optimizer to define learning rate
  optimizer = SGD(LM.parameters(), lr = lr, weight_decay = weight_decay)
  scheduler = StepLR(optimizer, step_size = lr_steprate, gamma = lr_gamma)

  return LM, optimizer, scheduler