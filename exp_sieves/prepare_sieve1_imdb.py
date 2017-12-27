"""
Prepares pickle output as a training dataset for Box-Filter
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

from util.model_util import makevar, printVec, get_n_params, get_box_feats, lamm, mbr, extract_order_feats
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

def run_filter(filter_net, box_rep, trees, boxes, golds, CNN, instance_idx, precision_k, annid2catid):

  prediction = filter_net([0]*5, box_rep, trees[instance_idx])
  pred_np    = prediction.cpu().data.numpy()

  if precision_k >= 1:
    precision_k = int(precision_k)
    f1_pred     = np.argsort(-pred_np)[0][:min(precision_k, pred_np.shape[1])]
  else:
    f1_pred = []
    for kk,prediction in enumerate(pred_np[0]):
      if np.exp(prediction) > precision_k:
        f1_pred.append(kk)
    f1_pred = np.array(f1_pred)

  gold_cats = set()
  for gold_box in golds[instance_idx]:
    ann_id = str(CNN['meta'][boxes[instance_idx][gold_box]])
    gold_cats.add(annid2catid[ann_id])

  cat_ids = [annid2catid[str(ann_id)] for ann_id in CNN['meta'][boxes[instance_idx]][f1_pred]]
  gold_cat   = [1 if cat_id in gold_cats else 0 for idx,cat_id in enumerate(cat_ids)]

  out_tuple = []
  box_list = list(np.array(boxes[instance_idx])[f1_pred])
  for ii,g in enumerate(gold_cat):
    out_tuple.append([trees[instance_idx],[box_list[ii]],[],g])
  return out_tuple

def filter_split(split, filter_net, CNN, precision_k, annid2catid):
  trees, boxes, ious, golds = split

  new_tree, new_box, new_iou, new_gold = [],[],[],[]
  for ii in xrange(len(trees)):
    box_feats, spat_feats = get_box_feats(boxes[ii], CNN)
    filter_box_rep = torch.cat((box_feats,spat_feats),1) ### assumes box usage == 0 for filter
    tuples = run_filter(filter_net, filter_box_rep, trees, boxes, golds, CNN, ii, precision_k, annid2catid)
    for t in tuples:
      new_tree.append(t[0])
      new_box.append(t[1])
      new_iou.append([])
      new_gold.append(t[3])

  return new_tree, new_box, new_iou, new_gold

args = get_baseline_args()
CNN  = h5py.File(args.cnn, 'r')
data = pickle.load(open(args.dump))

trn    = data['trn']
dev    = data['dev']
tst    = data['tst']
vocabs = data['vocabs']

if args.filter_model != '':
  filter_net = torch.load(args.filter_model, map_location=lambda storage, location: storage.cuda(0))
  filter_config = filter_net.config
  filter_net.eval()
  print "filter net is loaded!"
else:
  raise NotImplementedError()

if not os.path.exists(args.save_path):
  os.makedirs(args.save_path)
out_file = os.path.join(args.save_path, args.out_file)

print "loading meta data for box annotations(may not be necessary for all baselines)"
annid2imgid, annid2catid, annid2bsize, imgid2catid2bsize, imgid2catid2count = getCategoryDicts(args.refexp_root, args.coco_root)

print "preparing {}...".format('trn')
new_trn = filter_split(trn,filter_net, CNN, args.precision_k, annid2catid)
print "DONE with {} for {} instances".format('trn',len(new_trn[0]))

print "preparing {}...".format('dev')
new_dev = filter_split(dev,filter_net, CNN, args.precision_k, annid2catid)
print "DONE with {} for {} instances".format('dev',len(new_dev[0]))

print "preparing {}...".format('tst')
new_tst = filter_split(tst,filter_net, CNN, args.precision_k, annid2catid)
print "DONE with {} for {} instances".format('tst',len(new_tst[0]))

print "pickling..."
command = " ".join(sys.argv[1:])
pickle.dump({'trn' : new_trn,'dev' : new_dev, 'tst' : new_tst, 'vocabs' : vocabs, 'command' : command},open(out_file,'w'))
print "DONE with pickling! output at {}".format(out_file)
