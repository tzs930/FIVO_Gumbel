# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for creating sequence datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np
from scipy.sparse import coo_matrix
import tensorflow as tf
import torch

# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_pianoroll_to_dense(pianoroll, min_note, num_notes):
  """Converts a sparse pianoroll to a dense numpy array.

  Given a sparse pianoroll, converts it to a dense numpy array of shape
  [num_timesteps, num_notes] where entry i,j is 1.0 if note j is active on
  timestep i and 0.0 otherwise.

  Args:
    pianoroll: A sparse pianoroll object, a list of tuples where the i'th tuple
      contains the indices of the notes active at timestep i.
    min_note: The minimum note in the pianoroll, subtracted from all notes so
      that the minimum note becomes 0.
    num_notes: The number of possible different note indices, determines the
      second dimension of the resulting dense array.
  Returns:
    dense_pianoroll: A [num_timesteps, num_notes] numpy array of floats.
    num_timesteps: A python int, the number of timesteps in the pianoroll.
  """
  num_timesteps = len(pianoroll)
  inds = []
  for time, chord in enumerate(pianoroll):
    # Re-index the notes to start from min_note.
    inds.extend((time, note-min_note) for note in chord)
  shape = [num_timesteps, num_notes]
  values = [1.] * len(inds)
  sparse_pianoroll = coo_matrix(
      (values, ([x[0] for x in inds], [x[1] for x in inds])),
      shape=shape)
  return sparse_pianoroll.toarray(), num_timesteps

def create_pianoroll_dataset(path,
                             split,
                             batch_size,
                             num_parallel_calls=DEFAULT_PARALLELISM,
                             shuffle=False,
                             repeat=False,
                             min_note=21,
                             max_note=108):
  """Creates a pianoroll dataset.

  Args:
    path: The path of a pickle file containing the dataset to load.
    split: The split to use, can be train, test, or valid.
    batch_size: The batch size. If repeat is False then it is not guaranteed
      that the true batch size will match for all batches since batch_size
      may not necessarily evenly divide the number of elements.
    num_parallel_calls: The number of threads to use for parallel processing of
      the data.
    shuffle: If true, shuffles the order of the dataset.
    repeat: If true, repeats the dataset endlessly.
    min_note: The minimum note number of the dataset. For all pianoroll datasets
      the minimum note is number 21, and changing this affects the dimension of
      the data. This is useful mostly for testing.
    max_note: The maximum note number of the dataset. For all pianoroll datasets
      the maximum note is number 108, and changing this affects the dimension of
      the data. This is useful mostly for testing.
  Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, data_dimension]. The sequences in inputs are the
      sequences in targets shifted one timestep into the future, padded with
      zeros. This tensor is mean-centered, with the mean taken from the pickle
      file key 'train_mean'.
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, data_dimension].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
    mean: A float Tensor of shape [data_dimension] containing the mean loaded
      from the pickle file.
  """
  # Load the data from disk.
  num_notes = max_note - min_note + 1
  with open(path, "rb") as f:
    raw_data = pickle.load(f)

  pianorolls = raw_data[split]
  
#   mean = raw_data["train_mean"]
  num_examples = len(pianorolls) # [batch, seqlen, 88]

#   def pianoroll_generator():
#     for sparse_pianoroll in pianorolls:
#       yield sparse_pianoroll_to_dense(sparse_pianoroll, min_note, num_notes)
  dense_pianorolls = []  
  lengths = []

  for i, sparse_pianoroll in enumerate(pianorolls):
    dense_pianoroll, length = sparse_pianoroll_to_dense(sparse_pianoroll, min_note, num_notes)
    # target_dense_pianoroll, _ = sparse_pianoroll_to_dense(sparse_pianoroll[1:], min_note, num_notes)
    dense_pianorolls.append(dense_pianoroll)
    # dense_pianorolls_target.append(target_dense_pianoroll)
    lengths.append(length)

  maxlen = np.max(lengths)
  sources = []
  targets = [] 
  for i in range(len(dense_pianorolls)):
    sources.append(torch.tensor(dense_pianorolls[i][:-1], dtype=torch.float32).to(device))
    targets.append(torch.tensor(dense_pianorolls[i][1:], dtype=torch.float32).to(device))
  
  sources = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True).float().to(device)
  targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).float().to(device)
  lengths = torch.tensor(lengths)

  dataset = torch.utils.data.TensorDataset(sources, targets, lengths)
  dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
# dense_pianorolls_target.append(sparse_pianoroll_to_dense(sparse_pianoroll[:,], min_note, num_notes))
#   dataset = torch.utils.data.TensorDataset(dense_pianorolls, dense_pianorolls_target)
#   dataset = tf.data.Dataset.from_generator(
#       pianoroll_generator,
#       output_types=(tf.float64, tf.int64),
#       output_shapes=([None, num_notes], []))
#   if repeat: dataset = dataset.repeat()
#   if shuffle: dataset = dataset.shuffle(num_examples)

  # Batch sequences togther, padding them to a common length in time.
#   dataset = dataset.padded_batch(batch_size,
#                                  padded_shapes=([None, num_notes], []))

#   itr = dataset.make_one_shot_iterator()
#   inputs, targets, lengths = itr.get_next()
  return dataset_loader #, tf.constant(mean, dtype=tf.float32)


SQUARED_OBSERVATION = "squared"
ABS_OBSERVATION = "abs"
STANDARD_OBSERVATION = "standard"
OBSERVATION_TYPES = [SQUARED_OBSERVATION, ABS_OBSERVATION, STANDARD_OBSERVATION]

ROUND_TRANSITION = "round"
STANDARD_TRANSITION = "standard"
TRANSITION_TYPES = [ROUND_TRANSITION, STANDARD_TRANSITION]


def create_chain_graph_dataset(
    batch_size,
    num_timesteps,
    steps_per_observation=None,
    state_size=1,
    transition_variance=1.,
    observation_variance=1.,
    transition_type=STANDARD_TRANSITION,
    observation_type=STANDARD_OBSERVATION,
    fixed_observation=None,
    prefetch_buffer_size=2048,
    dtype="float32"):
  """Creates a toy chain graph dataset.

  Creates a dataset where the data are sampled from a diffusion process. The
  'latent' states of the process are sampled as a chain of Normals:

  z0 ~ N(0, transition_variance)
  z1 ~ N(transition_fn(z0), transition_variance)
  ...

  where transition_fn could be round z0 or pass it through unchanged.

  The observations are produced every steps_per_observation timesteps as a
  function of the latent zs. For example if steps_per_observation is 3 then the
  first observation will be produced as a function of z3:

  x1 ~ N(observation_fn(z3), observation_variance)

  where observation_fn could square z3, take the absolute value, or pass
  it through unchanged.

  Only the observations are returned.

  Args:
    batch_size: The batch size. The number of trajectories to run in parallel.
    num_timesteps: The length of the chain of latent states (i.e. the
      number of z's excluding z0.
    steps_per_observation: The number of latent states between each observation,
      must evenly divide num_timesteps.
    state_size: The size of the latent state and observation, must be a
      python int.
    transition_variance: The variance of the transition density.
    observation_variance: The variance of the observation density.
    transition_type: Must be one of "round" or "standard". "round" means that
      the transition density is centered at the rounded previous latent state.
      "standard" centers the transition density at the previous latent state,
      unchanged.
    observation_type: Must be one of "squared", "abs" or "standard". "squared"
      centers the observation density at the squared latent state. "abs"
      centers the observaiton density at the absolute value of the current
      latent state. "standard" centers the observation density at the current
      latent state.
    fixed_observation: If not None, fixes all observations to be a constant.
      Must be a scalar.
    prefetch_buffer_size: The size of the prefetch queues to use after reading
      and processing the raw data.
    dtype: A string convertible to a tensorflow datatype. The datatype used
      to represent the states and observations.
  Returns:
    observations: A batch of observations represented as a dense Tensor of
      shape [num_observations, batch_size, state_size]. num_observations is
      num_timesteps/steps_per_observation.
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch. Will contain num_observations as each entry.
  Raises:
    ValueError: Raised if steps_per_observation does not evenly divide
      num_timesteps.
  """
  if steps_per_observation is None:
    steps_per_observation = num_timesteps
  if num_timesteps % steps_per_observation != 0:
    raise ValueError("steps_per_observation must evenly divide num_timesteps.")
  num_observations = int(num_timesteps / steps_per_observation)
  def data_generator():
    """An infinite generator of latents and observations from the model."""
    transition_std = np.sqrt(transition_variance)
    observation_std = np.sqrt(observation_variance)
    while True:
      states = []
      observations = []
      # Sample z0 ~ Normal(0, sqrt(variance)).
      states.append(
          np.random.normal(size=[state_size],
                           scale=observation_std).astype(dtype))
      # Start the range at 1 because we've already generated z0.
      # The range ends at num_timesteps+1 because we want to include the
      # num_timesteps-th step.
      for t in range(1, num_timesteps+1):
        if transition_type == ROUND_TRANSITION:
          loc = np.round(states[-1])
        elif transition_type == STANDARD_TRANSITION:
          loc = states[-1]
        z_t = np.random.normal(size=[state_size], loc=loc, scale=transition_std)
        states.append(z_t.astype(dtype))
        if t % steps_per_observation == 0:
          if fixed_observation is None:
            if observation_type == SQUARED_OBSERVATION:
              loc = np.square(states[-1])
            elif observation_type == ABS_OBSERVATION:
              loc = np.abs(states[-1])
            elif observation_type == STANDARD_OBSERVATION:
              loc = states[-1]
            x_t = np.random.normal(size=[state_size],
                                   loc=loc,
                                   scale=observation_std).astype(dtype)
          else:
            x_t = np.ones([state_size]) * fixed_observation

          observations.append(x_t)
      yield states, observations

  dataset = tf.data.Dataset.from_generator(
      data_generator,
      output_types=(tf.as_dtype(dtype), tf.as_dtype(dtype)),
      output_shapes=([num_timesteps+1, state_size],
                     [num_observations, state_size])
  )
  dataset = dataset.repeat().batch(batch_size)
  dataset = dataset.prefetch(prefetch_buffer_size)
  itr = dataset.make_one_shot_iterator()
  _, observations = itr.get_next()
  # Transpose observations from [batch, time, state_size] to
  # [time, batch, state_size].
  observations = tf.transpose(observations, perm=[1, 0, 2])
  lengths = tf.ones([batch_size], dtype=tf.int32) * num_observations
  return observations, lengths
