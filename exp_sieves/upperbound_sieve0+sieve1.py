"""
boxes -> sieve 0 -> sieve 1 -> oracle (upperbound)

"""
__author__ = "volkan cirik"

import random, sys, os, h5py, json

import numpy as np
import cPickle as pickle
from tqdm import tqdm

import torch
import torch.nn as nn

from util.model_util import makevar, get_box_feats
from util.arguments import get_baseline_args
from collections import defaultdict

def run_filter(filter_net, box_filter_net , tree, boxes, w2i, CNN, precision_f1, instance_idx, gold):
  box_feats,spat_feats = get_box_feats(boxes[instance_idx], CNN)
  box_rep = torch.cat((box_feats,spat_feats),1)

  prediction = filter_net([0]*5, box_rep, tree)
  pred_np    = prediction.cpu().data.numpy()

  if precision_f1 >= 1:
    precision_f1 = int(precision_f1)
    f1_pred     = np.argsort(-pred_np)[0][:min(precision_f1, pred_np.shape[1])]
  else:
    f1_pred = []
    for kk,prediction in enumerate(pred_np[0]):
      if np.exp(prediction) > precision_f1:
        f1_pred.append(kk)
    f1_pred = np.array(f1_pred)

  bf_pred        = []
  for box_index in f1_pred:
    box_feats,spat_feats = get_box_feats([boxes[instance_idx][box_index]], CNN)
    box_rep = torch.cat((box_feats,spat_feats),1) ### assumes box_usage = 0 fir box_filter_net
    prediction = box_filter_net([w2i.get(node.label,0) for node in tree.leaves()], box_rep, tree).data
    bf_pred.append(prediction[0][0])

  return bf_pred, f1_pred

def evaluate(filter_net, box_filter_net, split, CNN, verbose = False, precision_f1 = 4, split_name = 'trn', out_file = 'out.csv'):

  trees, boxes, iou, gold = split

  indexes = range(len(trees))
  if verbose:
    pbar = tqdm(indexes)
  else:
    pbar = indexes

  THRESHOLD = [0.01,0.05,0.1,0.2]
  RANKS     = range(2,6)

  th_count  = { th:defaultdict(int) for th in THRESHOLD}
  rd_count  = { rank:defaultdict(int) for rank in RANKS}

  hit_th = [0.0]*len(THRESHOLD)
  hit_rd = [0.0]*len(RANKS)
  total  = 0.0
  for j in pbar:
    tree = trees[j]
    bf_pred ,box_indexes = run_filter(filter_net, box_filter_net, tree, boxes, w2i, CNN, precision_f1, j, gold)

    for ii,th in enumerate(THRESHOLD):
      threshold_pred = []
      for kk,prediction in enumerate(bf_pred):
        if prediction > th:
          threshold_pred.append(box_indexes[kk])
      th_count[th][len(threshold_pred)] += 1.0
      if len(set(threshold_pred).intersection(set(gold[j]))) > 0:
        hit_th[ii] += 1.0

    for ii,rank in enumerate(RANKS):
      ranked_pred = box_indexes[np.argsort(-np.array(bf_pred))[:min(rank, len(bf_pred))]]
      rd_count[rank][len(ranked_pred)] += 1.0
      if len(set(ranked_pred).intersection(set(gold[j]))) > 0:
        hit_rd[ii] += 1.0
    total += 1.0

  out = open(out_file, 'a+')
  f1_config = "p@{}".format(int(precision_f1)) if precision_f1 >= 1 else "th>={}".format(precision_f1)
  print "_"*20
  print "split: {} filter1 {}".format(split_name, f1_config)
  for ii,th in enumerate(THRESHOLD):
    print "threshold: {} th_hit: {:5.3f} avg {}".format(th, hit_th[ii]/total, sum([th_count[th][count]*count for count in th_count[th]])/total)
    f2_config = "th>={}".format(th)
    config    = "f1-{}_f2-{}".format(f1_config,f2_config)
    print >> out, "{},{},{}".format(config, hit_th[ii]/total, sum([th_count[th][count]*count for count in th_count[th]])/total)

  for ii,rank in enumerate(RANKS):
    print "rank: {} rd_hit: {:5.3f} avg {}".format(rank, hit_rd[ii]/total, sum([rd_count[rank][count]*count for count in rd_count[rank]])/total)
    f2_config = "p@{}".format(rank)
    config    = "f1-{}_f2-{}".format(f1_config,f2_config)
    print >> out, "{},{},{}".format(config, hit_rd[ii]/total, sum([rd_count[rank][count]*count for count in rd_count[rank]])/total)
#    print "rank: {} rd_hit: {:5.3f} dist {} avg {}".format(rank, hit_rd[ii]/total," ".join([str(rd_count[rank][count]) for count in rd_count[rank]]), sum([rd_count[rank][count]*count for count in rd_count[rank]])/total)

args = get_baseline_args()
CNN  = h5py.File(args.cnn, 'r')
data = pickle.load(open(args.dump))

trn     = data['trn']
dev     = data['dev']
tst     = data['tst']
vocabs  = data['vocabs']

vectors, w2i, p2i, n2i, i2w, i2p, i2n = vocabs

if args.filter_model != '':
  filter_net = torch.load(args.filter_model, map_location=lambda storage, location: storage.cuda(0))
  filter_config = filter_net.config
  filter_net.eval()
  print "filter net is loaded!"
else:
  raise NotImplementedError()
if args.box_filter_model != '':
  box_filter_net = torch.load(args.box_filter_model, map_location=lambda storage, location: storage.cuda(0))
  box_filter_config = box_filter_net.config
  box_filter_net.eval()
  print "box_filter net is loaded!"
else:
  raise NotImplementedError()

for precision_f1 in [0.01,0.05,0.1,0.2] + range(2,6):
  evaluate(filter_net, box_filter_net, dev, CNN, precision_f1 = precision_f1, verbose = True, split_name = 'dev', out_file = args.out_file)
