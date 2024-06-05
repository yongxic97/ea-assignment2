from __future__ import annotations
import numpy as np
import torch.nn as nn
import torch
from genepro.node_impl import *

class Multitree(nn.Module):
  def __init__(self, n_trees: int):
    super(Multitree, self).__init__()
    self.n_trees = n_trees
    self.children = []

  def get_output_pt(self, x):
    output = []
    for child in self.children:
      output.append(child.get_output_pt(x).view(-1,1))

    return torch.cat(output,dim=1)

  def get_subtrees_consts(self):
    constants = []
    for child in self.children:
      constants.extend([node.pt_value for node in child.get_subtree() if isinstance(node, Constant)])
    return constants

  def __len__(self) -> int:
    """
    Returns the max length of the trees in the multi-tree
    """
    lens = [len(child) for child in self.children]
    return np.max(lens)

  def get_readable_repr(self) -> str:
    return [child.get_readable_repr() for child in self.children]
  
  def optimise_coefficients(self, batch, loss_fn, optimiser_type, n_epochs,gamma=0.9, learn=1e-3):
    constants = self.get_subtrees_consts()
    if len(constants) == 0:
      raise Exception("No constants in the tree")
    optimiser = optimiser_type(constants, lr=learn, amsgrad=True)
    for epoch in range(n_epochs):
      state = torch.cat(batch.state)
      action = torch.cat(batch.action)
      reward = torch.cat(batch.reward)
      non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), dtype=torch.bool)

      non_final_next_states = torch.cat([s for s in batch.next_state
                                              if s is not None])
      output = self.get_output_pt(state).gather(1, action)

      next_state_values= torch.zeros(reward.shape[0], dtype=torch.float)
      with torch.no_grad():
        next_state_values[non_final_mask] = self.get_output_pt(non_final_next_states).max(dim=1)[0].float()
      target = reward+gamma*next_state_values
      loss = loss_fn(output, target.unsqueeze(1).detach())



      optimiser.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_value_(constants, 100)
      optimiser.step()
