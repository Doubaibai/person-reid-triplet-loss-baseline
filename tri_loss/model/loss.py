from __future__ import print_function
import torch
from torch.autograd import Variable

def normalize(x, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input      
  """
  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x


def euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist

def euclidean_dist_div(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d, 4]
    y: pytorch Variable, with shape [n, d, 4]
  Returns:
    dist: pytorch Variable, with shape [m *4, n *4]
  """
  m, n = x.size(0), y.size(0)
  k = x.size(2)-1
  x, x1,x2,x3,x4 = x.chunk(k+1, dim=2)
  y, y1,y2,y3,y4 = y.chunk(k+1, dim=2)
  x_all = torch.cat((x1.squeeze(), x2.squeeze(), x3.squeeze(), x4.squeeze()), 0)
  
  # y = torch.cat((y1.squeeze(), y2.squeeze(), y3.squeeze(), y4.squeeze()), 0)
  # x_all m*4, d
  # y n, d

  dummy_eye = torch.eye(m).repeat(k, 1).cuda()
  xx = torch.pow(x_all, 2).sum(1, keepdim=True).expand(m*k, n)
  yy = torch.pow(y.squeeze(), 2).sum(1, keepdim=True).expand(n, m*k).t()
  dist = xx + yy
  dist.addmm_(1, -2, x_all, y.squeeze().t())
  # dist = dist + 10*dummy_eye
  dist = torch.add(dist ,10, Variable(dummy_eye))
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist

# def hard_example_mining(dist_mat, labels, return_inds=False):
#   """For each anchor, find the hardest positive and negative sample.
#   Args:
#     dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#     labels: pytorch LongTensor, with shape [N]
#     return_inds: whether to return the indices. Save time if `False`(?)
#   Returns:
#     dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#     dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#     p_inds: pytorch LongTensor, with shape [N]; 
#       indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#     n_inds: pytorch LongTensor, with shape [N];
#       indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#   NOTE: Only consider the case in which all labels have same num of samples, 
#     thus we can cope with all anchors in parallel.
#   """

#   assert len(dist_mat.size()) == 2
#   assert dist_mat.size(0) == dist_mat.size(1)
#   N = dist_mat.size(0)

#   # shape [N, N]
#   is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#   is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

#   # `dist_ap` means distance(anchor, positive)
#   # both `dist_ap` and `relative_p_inds` with shape [N, 1]
#   dist_ap, relative_p_inds = torch.max(
#     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
#   # `dist_an` means distance(anchor, negative)
#   # both `dist_an` and `relative_n_inds` with shape [N, 1]
#   dist_an, relative_n_inds = torch.min(
#     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#   # shape [N]
#   dist_ap = dist_ap.squeeze(1)
#   dist_an = dist_an.squeeze(1)

#   if return_inds:
#     # shape [N, N]
#     ind = (labels.new().resize_as_(labels)
#            .copy_(torch.arange(0, N).long())
#            .unsqueeze( 0).expand(N, N))
#     # shape [N, 1]
#     p_inds = torch.gather(
#       ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
#     n_inds = torch.gather(
#       ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
#     # shape [N]
#     p_inds = p_inds.squeeze(1)
#     n_inds = n_inds.squeeze(1)
#     return dist_ap, dist_an, p_inds, n_inds

#   return dist_ap, dist_an


# def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
#   """
#   Args:
#     tri_loss: a `TripletLoss` object
#     global_feat: pytorch Variable, shape [N, C]
#     labels: pytorch LongTensor, with shape [N]
#     normalize_feature: whether to normalize feature to unit length along the 
#       Channel dimension
#   Returns:
#     loss: pytorch Variable, with shape [1]
#     p_inds: pytorch LongTensor, with shape [N]; 
#       indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#     n_inds: pytorch LongTensor, with shape [N];
#       indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     ==================
#     For Debugging, etc
#     ==================
#     dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#     dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#     dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
#   """
#   if normalize_feature:
#     global_feat = normalize(global_feat, axis=-1)
#   # shape [N, N]
#   dist_mat = euclidean_dist(global_feat, global_feat)
#   dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
#     dist_mat, labels, return_inds=True)
#   loss = tri_loss(dist_ap, dist_an)
#   return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat

def hard_example_mining(dist_mat, labels, return_inds=False):
  """For each anchor, find the hardest positive and negative sample.
  Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N*4, N*4]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
  Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N]; 
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
  NOTE: Only consider the case in which all labels have same num of samples, 
    thus we can cope with all anchors in parallel.
  """

  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == 4 * dist_mat.size(1)
  N = dist_mat.size(1)

  # shape [N, N]
  is_pos = (labels.expand(N, N).eq(labels.expand(N, N).t())-torch.eye(N).byte().cuda()).repeat(1, 4)
  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).repeat(1, 4)

  # transform dist_mat from N*4, N to N, N*4
  dist_ms = torch.split(dist_mat, N, dim=0)
  dist_mat = torch.cat(dist_ms, 1)

  # `dist_ap` means distance(anchor, positive)
  # both `dist_ap` and `relative_p_inds` with shape [N, 1]
  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  # dis_rec means distance of reconstruction
  dist_rec, relative_rec_inds = torch.min(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  # `dist_an` means distance(anchor, negative)
  # both `dist_an` and `relative_n_inds` with shape [N, 1]
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)
  dist_rec = dist_rec.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (labels.new().resize_(labels.size()[0]*4 )
           .copy_(torch.arange(0, N*4).long())
           .unsqueeze( 0).expand(N, N*4))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    rec_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_rec_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    rec_inds = rec_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, dist_rec, p_inds, n_inds, rec_inds

  return dist_ap, dist_an, dist_rec

def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
  """
  Args:
    tri_loss: a `TripletLoss` object
    global_feat: pytorch Variable, shape [N, C]
    labels: pytorch LongTensor, with shape [N]
    normalize_feature: whether to normalize feature to unit length along the 
      Channel dimension
  Returns:
    loss: pytorch Variable, with shape [1]
    p_inds: pytorch LongTensor, with shape [N]; 
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    ==================
    For Debugging, etc
    ==================
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
  """
  if normalize_feature:
    global_feat = normalize(global_feat, axis=-1)
  # shape [N, N]
  dist_mat = euclidean_dist_div(global_feat, global_feat)
  dist_ap, dist_an, dist_rec, p_inds, n_inds, rec_inds = hard_example_mining(
    dist_mat, labels, return_inds=True)
  loss = tri_loss(dist_ap, dist_an) + dist_rec.norm()
  return loss, p_inds, n_inds, rec_inds, dist_ap, dist_an, dist_rec, dist_mat