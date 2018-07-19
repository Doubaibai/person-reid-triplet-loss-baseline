"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    m, n = array1.shape[0], array2.shape[0]
    k = array1.shape[2]-1
    x_list = np.split(array1, k+1, axis=2)
    y, y1,y2,y3,y4 = np.split(array2, k+1, axis=2)

    # x_all = torch.cat((x1.squeeze(), x2.squeeze(), x3.squeeze(), x4.squeeze()), 1)
    for i, x in enumerate(x_list):
      if i == 0:
        continue
      square1 = np.sum(np.square(x.squeeze()), axis=1)[..., np.newaxis]
      # shape [1, m2]
      square2 = np.sum(np.square(y.squeeze()), axis=1)[np.newaxis, ...]
      squared_dist = - 2 * np.matmul(x.squeeze(), y.squeeze().T) + square1 + square2
      squared_dist[squared_dist < 0] = 0
      dist = np.sqrt(squared_dist)
      if i == 1:
      	dist_final = dist
      else:
      	dist_final = np.minimum(dist_final, dist)
    return dist_final

# def compute_dist(array1, array2, type='euclidean'):
#   """Compute the euclidean or cosine distance of all pairs.
#   Args:
#     array1: numpy array with shape [m1, n]
#     array2: numpy array with shape [m2, n]
#     type: one of ['cosine', 'euclidean']
#   Returns:
#     numpy array with shape [m1, m2]
#   """
#   import pdb
#   pdb.set_trace()
#   assert type in ['cosine', 'euclidean']
#   if type == 'cosine':
#     array1 = normalize(array1, axis=1)
#     array2 = normalize(array2, axis=1)
#     dist = np.matmul(array1, array2.T)
#     return dist
#   else:
#     # shape [m1, 1]
#     square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
#     # shape [1, m2]
#     square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
#     squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
#     squared_dist[squared_dist < 0] = 0
#     dist = np.sqrt(squared_dist)
#     return dist
