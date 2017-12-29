"""
boxes -> sieve 0 -> sieve 1 -> sieve 2 {cmn_loc,cmn}
"""
__author__ = "volkan cirik"

import time
import math
start = time.time()

import random, sys, os, h5py, json

import numpy as np
import cPickle as pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from util.model_util import makevar, printVec, get_n_params, get_box_feats, lamm, mbr
from models.get_model import get_model
from util.arguments import get_baseline_args
from collections import OrderedDict, defaultdict

def run_filter(filter_net, box_filter_net, tree, boxes, w2i, CNN, gold, instance_idx, precision_f1, precision_f2):
  box_feats,spat_feats = get_box_feats(boxes[instance_idx], CNN)
  box_rep = torch.cat((box_feats,spat_feats),1)

  prediction = filter_net([0]*5, box_rep, tree)
  pred_np    = prediction.cpu().data.numpy()

  if precision_f1 >= 1:
    precision_f1 = int(precision_f1)
    f1_pred      = np.argsort(-pred_np)[0][:min(precision_f1, pred_np.shape[1])]
  else:
    f1_pred = []
    for kk,prediction in enumerate(pred_np[0]):
      if np.exp(prediction) > precision_f1:
        f1_pred.append(kk)
    f1_pred = np.array(f1_pred)

  bf_pred        = []
  for box_index in f1_pred:
    box_feats,spat_feats = get_box_feats([boxes[instance_idx][box_index]], CNN)
    box_rep = torch.cat((box_feats,spat_feats),1) ### assumes box_usage = 0 for box_filter_net
    prediction = box_filter_net([w2i.get(node.label,0) for node in tree.leaves()], box_rep, tree).data
    bf_pred.append(prediction[0][0])

  if precision_f2 >= 1:
    precision_f2 = int(precision_f2)
    final_prediction = f1_pred[np.argsort(-np.array(bf_pred))[:min(precision_f2, len(bf_pred))]]
  else:
    final_prediction = []
    for kk,pred in enumerate(bf_pred):
      if pred > precision_f2:
        final_prediction.append(f1_pred[kk])
    final_prediction = np.array(final_prediction)
  return final_prediction, len(set(final_prediction).intersection(set(gold[instance_idx])))

def evaluate(filter_net, box_filter_net, net, split, CNN, config, verbose = False, tst_json = [], out_file = '', precision_f1 = 5, precision_f2 = 5, no_lang = False):

  box_usage = config['box_usage']
  model     = config['model']

  eval_start = time.time()
  n = correct = 0.0
  trees, boxes, iou, gold = split

  indexes = range(len(trees))
  if verbose:
    pbar = tqdm(indexes)
  else:
    pbar = indexes

  preds = []
  net.eval()

  all_supporting = []

  for j in pbar:
    tree = trees[j]
    filter_pred, gold_predicted = run_filter(filter_net, box_filter_net, tree, boxes, w2i, CNN, gold, j, precision_f1, precision_f2)
    if gold_predicted == 0:
      n += 1
      preds.append([[-1,-1,-1,-1]])
      all_supporting.append([])
      continue

    gold_instance = []
    for g in gold[j]:
      if g in set(filter_pred):
        gold_instance.append((filter_pred == g).nonzero()[0][0])

    box_feats, spat_feats = get_box_feats(list(np.array(boxes[j])[filter_pred]), CNN)

    if box_usage == 0:
      box_rep = torch.cat((box_feats,spat_feats),1)
    elif box_usage == 1:
      box_rep = box_feats
    elif box_usage == 2:
      box_rep = spat_feats
    else:
      raise NotImplementedError()
    prediction  = net([w2i.get(node.label,0) for node in tree.leaves()], box_rep, tree)
    supporting  = []
    _,pred = torch.max(prediction.data,1)
    hit        = (1.0 if pred[0][0] in set(gold_instance) else 0.0)
    correct   += hit
    n += 1
    if len(tst_json) > 0:
      pred_np  = prediction.cpu().data.numpy()
      preds.append(np.array(tst_json[j]['box_names'])[np.argsort(-pred_np)[0]])
    all_supporting.append(supporting)
  eval_time = time.time() - eval_start

  if tst_json != []:
    for ii,inst in enumerate(tst_json):
      tst_json[ii]['predicted_bounding_boxes'] = [list(p) for p in preds[ii]]
      tst_json[ii]['context_box'] = all_supporting[ii]
    json.dump(tst_json,open(out_file,'w'))

  return correct/n , len(trees)/eval_time

args = get_baseline_args()
CNN  = h5py.File(args.cnn, 'r')
data = pickle.load(open(args.dump))

trn     = data['trn'] 
dev     = data['dev']
tst     = data['tst']
vocabs  = data['vocabs']
tst_json= data['tst_json']

vectors, w2i, p2i, n2i, i2w, i2p, i2n = vocabs
Xtrn_tree, Xtrn_box, Xtrn_iou, Ytrn   = trn

indexes = range(len(Xtrn_tree))

if args.box_usage == 0:
  feat_box = 4096 + 5
elif args.box_usage == 1:
  feat_box = 4096
elif args.box_usage == 2:
  feat_box = 5
else:
  raise NotImplementedError()

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

if args.resume != '':
  net = torch.load(args.resume, map_location=lambda storage, location: storage.cuda(0))
  config = net.config
  precision_f1 = net.config['precision_f1']
  precision_f2 = net.config['precision_f2']
  net.debug = args.debug_mode
else:
  raise NotImplementedError()

if args.out_file == '':
  raise NotImplementedError()
else:
  print("will dump prediction to {}".format(args.out_file))

best_val, val_rate  = evaluate(filter_net, box_filter_net, net, dev, CNN, config, verbose = args.verbose, precision_f1 = args.precision_f1, precision_f2 = args.precision_f2)
tst_score, tst_rate = evaluate(filter_net, box_filter_net, net, tst, CNN, config, verbose = args.verbose, precision_f1 = args.precision_f1, precision_f2 = args.precision_f2, tst_json = tst_json, out_file = args.out_file)
print "\nmodel scores based on best validation accuracy with precision_f1 {} and precision_f2 {}\nval_acc:{:5.3f} test_acc: {:5.3f} test speed {:5.1f} inst/sec\n".format(precision_f1,precision_f2,best_val,tst_score,tst_rate)
