#from __future__ import absolute_import
import torch

from models import cmn, cmn_loc, box_filter
from util.model_util import *

def get_model(vocabs, config):
  vectors, w2i, p2i, n2i, i2w, i2p, i2n = vocabs
  config['word_dim'] = vectors.shape[1]
  if config['model'] == 'cmn':
    net = cmn.CMN(w2i, p2i, n2i, config)
  elif config['model'] == 'cmn_loc':
    net = cmn_loc.CMN_LOC(w2i, p2i, n2i, config)
  elif config['model'] == 'box_filter':
    net = box_filter.BOX_FILTER(w2i, p2i, n2i, config)
  else:
    raise NotImplementedError()

  net.Wwrd.weight.data.copy_(torch.from_numpy(vectors).cuda())
  if not config['finetune']:
    net.Wwrd.weight.requires_grad=False
  net.cuda()
  for param in net.parameters():
    weight_init(param)
  return net
