"""
boxes -> sieve 0 -> oracle for gold category boxes -> {cmn_loc,cmn}
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

def getCategoryDicts(refexp_root, coco_root):

  refexp_anns = ['{}/google_refexp_val_201511_coco_aligned_mcg_umd.json'.format(refexp_root),'{}/google_refexp_train_201511_coco_aligned_mcg_umd.json'.format(refexp_root)]
  coco_anns = ['{}/instances_train2014.json'.format(coco_root),'{}/instances_val2014.json'.format(coco_root)]

  annid2imgid = {}
  imgid2catid2count = {}
  imgid2catid2bsize = {}
  annid2catid = {}
  annid2bsize = {}

  for refexp_ann in refexp_anns:
    refexp = json.load(open(refexp_ann))
    for ann_id in refexp['annotations']:
      inst = refexp['annotations'][ann_id]
      annid2imgid[str(inst['annotation_id'])] = str(inst['image_id'])
      annid2catid[str(inst['annotation_id'])] = str(inst['category_id'])
      annid2bsize[str(inst['annotation_id'])] = int(inst['bbox'][-2]*inst['bbox'][-1])
  for coco_ann in coco_anns:
    coco = json.load(open(coco_ann))
    for inst in coco['annotations']:
      catid = str(inst['category_id'])
      id = str(inst['id'])
      annid2catid[id] = catid

  return annid2imgid, annid2catid, annid2bsize, imgid2catid2bsize, imgid2catid2count


def run_filter(filter_net, tree, boxes, CNN, gold, instance_idx, precision_f1, annid2catid):
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

  gold_cats = set()
  for gold_box in gold[instance_idx]:
    ann_id = str(CNN['meta'][boxes[instance_idx][gold_box]])
    gold_cats.add(annid2catid[ann_id])

  cat_ids = [annid2catid[str(ann_id)] for ann_id in CNN['meta'][boxes[instance_idx]][f1_pred]]
  filtered_cat_ids = [idx for idx,cat_id in enumerate(cat_ids) if cat_id in gold_cats]
  final_pred = f1_pred[filtered_cat_ids] ### filter by obj category

  return final_pred, len(set(final_pred).intersection(set(gold[instance_idx])))

def evaluate(filter_net,net, split, CNN, config, precision_f1,
             verbose = False, tst_json = [], out_file = '', annid2catid = {}):

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
    filter_pred, gold_predicted = run_filter(filter_net, tree, boxes, CNN, gold, j, precision_f1, annid2catid)

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
    supporting = []
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

if args.resume != '':
  net = torch.load(args.resume, map_location=lambda storage, location: storage.cuda(0))
  config = net.config
  precision_f1 = net.config['precision_f1']
  net.debug = args.debug_mode
else:
  raise NotImplementedError()

if args.out_file == '':
  raise NotImplementedError()
else:
  print("will dump prediction to {}".format(args.out_file))

print "loading meta data for box annotations(may not be necessary for all baselines)"
annid2imgid, annid2catid, annid2bsize, imgid2catid2bsize, imgid2catid2count = getCategoryDicts(args.refexp_root, args.coco_root)

best_val, val_rate = evaluate(filter_net, net, dev, CNN, config, precision_f1,
                              verbose = args.verbose, annid2catid = annid2catid)
tst_score, tst_rate = evaluate(filter_net, net, tst, CNN, config, precision_f1,
                               verbose = args.verbose, tst_json = tst_json,
                               out_file = args.out_file, annid2catid = annid2catid)
print "\nmodel scores based on best validation accuracy with precision_f1 {}\nval_acc:{:5.3f} test_acc: {:5.3f} test speed {:5.1f} inst/sec\n".format(precision_f1,best_val,tst_score,tst_rate)
