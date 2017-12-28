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

def evaluate(filter_net, box_filter_net, net, split, CNN, config, experiment_log,
             box_usage = 0, verbose = False, tst_json = [],
             out_file = '', precision_f1 = 5, precision_f2 = 5, no_lang = False):

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
      preds.append(None)
      all_supporting.append(None)
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
    preds.append(filter_pred[int(pred[0][0])])
    all_supporting.append(supporting)
  eval_time = time.time() - eval_start

  if tst_json != []:
    for ii,inst in enumerate(tst_json):
      if preds[ii] == None:
        tst_json[ii]['predicted_bounding_boxes'] = [[-1,-1,-1,-1]]
        tst_json[ii]['context_box'] = [-1]
        continue
      tst_json[ii]['predicted_bounding_boxes'] = [tst_json[ii]['box_names'][preds[ii]]]
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
config = OrderedDict([('model', args.model),
                      ('n_hidden' , args.n_hidden),
                      ('dropout' , args.dropout),
                      ('fusion' , args.fusion),
                      ('feat_box' , feat_box),
                      ('precision_f1', args.precision_f1),
                      ('precision_f2', args.precision_f2),
                      ('clip' , args.clip),
                      ('finetune' , args.finetune),
                      ('use_outer' , args.use_outer),
                      ('box_usage', args.box_usage),
                      ('loss' , args.loss),
                      ('optim' , args.optim),
                      ('lr' , args.lr),
                      ('lr_min' , args.lr_min),
                      ('lr_decay' , args.lr_decay),
                      ('weight_decay' , args.weight_decay),
                      ('debug'  , args.debug_mode)])
net = get_model(vocabs, config)

if not os.path.exists(args.save_path):
  os.makedirs(args.save_path)
snapshot_pfx   = 'snapshot.' + ".".join([key.upper()+str(config[key]) for key in config.keys() if key[0] != 'f'])
snapshot_model = os.path.join(args.save_path, snapshot_pfx + '.model')
experiment_log = open(os.path.join(args.save_path, snapshot_pfx + '.log'),'w')

out_file = os.path.join(args.save_path, snapshot_pfx + '.tst-eval.json')
print("="*20)
print("Starting training {} model".format(config['model']))
print("Snapshots {}.*\nDetails:".format(os.path.join(args.save_path, snapshot_pfx)))
for key in config:
  print("{} : {}".format(key,config[key]))
print("="*20)
if config['optim'] == 'sgd':
  optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = config['lr'], momentum = 0.95)
else:
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr = config['lr'], weight_decay = config['weight_decay'])
optimizer.zero_grad()

if config['loss'] == 'nll':
  criterion = nn.NLLLoss()
elif config['loss'] == 'smargin':
  criterion = nn.MultiLabelSoftMarginLoss()
elif config['loss'] == 'lamm':
  criterion = lamm
elif config['loss'] == 'mbr':
  criterion = mbr
else:
  raise NotImplementedError()

start_time = time.time()
if args.verbose:
  print("startup time for {} model: {:5.3f} for {} instances for {} parameters".format(config['model'],start_time - start, len(indexes), get_n_params(net)))

best_val = 0
timeout = False
for ITER in range(args.epochs):
  net.train()
  random.shuffle(indexes)
  closs = 0.0
  cinst = 0
  correct = 0.0
  trn_start = time.time()

  if args.verbose and not args.debug_mode:
    pbar = tqdm(indexes, desc='trn_loss')
  else:
    pbar = indexes

  done = 1
  for ii in pbar:
    tree = Xtrn_tree[ii]
    filter_pred, gold_predicted = run_filter(filter_net, box_filter_net, tree, Xtrn_box, w2i, CNN, Ytrn, ii, args.precision_f1, args.precision_f2)
    if gold_predicted == 0:
      cinst += 1
      continue

    gold_instance = []
    for g in Ytrn[ii]:
      if g in set(filter_pred):
        gold_instance.append((filter_pred == g).nonzero()[0][0])

    box_feats, spat_feats = get_box_feats(list(np.array(Xtrn_box[ii])[filter_pred]), CNN)

    if config['box_usage'] == 0:
      box_rep = torch.cat((box_feats,spat_feats),1)
    elif config['box_usage'] == 1:
      box_rep = box_feats
    elif config['box_usage'] == 2:
      box_rep = spat_feats
    else:
      raise NotImplementedError()

    if args.debug_mode:
      raise NotImplementedError()
    else:
      prediction = net([w2i.get(n.label,0) for n in tree.leaves()], box_rep, tree)
      _,pred = torch.max(prediction.data,1)
      correct += (1.0 if pred[0][0] in set(gold_instance) else 0.0)

      if config['loss'] == 'nll':
        gold = makevar(gold_instance[0])
      elif config['loss'] == 'smargin':
        gold = np.zeros((1,box_rep.size(0)))
        np.put(gold,gold_instance,1.0)
        gold = makevar(gold, numpy_var = True).view(1,box_rep.size(0))
      else:
        raise NotImplementedError()

      loss  = criterion(prediction, gold)
      closs += float(loss.data[0])
      cinst += 1
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm(net.parameters(), config['clip'])
      optimizer.step()

      if args.verbose:
        pbar.set_description("trn_loss {:5.3f} trn_acc {:5.3f}".format(closs/cinst,correct/cinst))
      if time.time() - start_time > args.timeout:
        timeout = True
        break

    if timeout:
      break
    if args.val_freq != 0 and done % args.val_freq == 0:
      print("")
      val_score, val_rate = evaluate(filter_net, box_filter_net, dev, CNN, config, experiment_log, verbose = False, precision_f1 = args.precision_f1, precision_f2 = args.precision_f2)
      print("epoch {:3d}/{:3d} inst#{:3d} val_acc: {:5.3f}".format(ITER+1,args.epochs,ii,val_score))
    done += 1
  trn_loss = closs / cinst
  trn_acc  = correct/cinst
  trn_rate = len(indexes)/(time.time() - trn_start)
  val_score, val_rate = evaluate(filter_net, box_filter_net, net, dev, CNN, config, experiment_log, verbose = args.verbose, precision_f1 = args.precision_f1, precision_f2 = args.precision_f2)

  log = "epoch {:3d}/{:3d} trn_loss: {:5.3f} trn_acc: {:5.3f} trn speed {:5.1f} inst/sec \n\t\tbest_val: {:5.3f} val_acc: {:5.3f} val speed {:5.1f} inst/sec\n".format(ITER+1,args.epochs,trn_loss,trn_acc,trn_rate,best_val,val_score,val_rate)
  config['lr'] = max(config['lr']*config['lr_decay'],config['lr_min'])

  if config['optim'] == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = config['lr'], momentum = 0.95, weight_decay = config['weight_decay'])
  else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr = config['lr'], weight_decay = config['weight_decay'])

  if args.verbose:
    print(log)
    print("lr is updated to {}".format(config['lr']))
  experiment_log.write(log)
  torch.save(net, snapshot_model + '.EPOCH' + str(ITER+1))

  if best_val < val_score:
    if args.verbose:
      print("Best model is updated at epoch {}".format(ITER+1))
    torch.save(net, snapshot_model + '.best')
    best_val = val_score
  experiment_log.flush()
  if args.yo:
    stats = "epoch {:3d}/{:3d}-trn_loss: {:5.3f}-trn_acc: {:5.3f}-trn speed {:5.1f} inst/sec-best_val: {:5.3f}-val_acc: {:5.3f}-val speed {:5.1f} inst/sec".format(ITER+1,args.epochs,trn_loss,trn_acc,trn_rate,best_val,val_score,val_rate).split('-')
    text  = "\n".join(["{} : {}".format(arg,getattr(args,arg)) for arg in vars(args)] + stats)
    response = requests.post('http://api.justyo.co/yo/', data={'api_token': YO_API_TOKEN,'text' : text, 'username' : YO_USERNAME})
    print "YO! RESPONSE:",response

best_net = torch.load(snapshot_model + '.best', map_location=lambda storage, location: storage.cuda(0))
tst_score, tst_rate = evaluate(filter_net, box_filter_net, best_net, tst, CNN, config, experiment_log,
                               verbose = args.verbose, tst_json = tst_json,
                               out_file = out_file, precision_f1 = args.precision_f1, precision_f2 = args.precision_f2)
log = "model scores based on best validation accuracy\nval_acc:{:5.3f} test_acc: {:5.3f} test speed {:5.1f} inst/sec\n".format(best_val,tst_score,tst_rate)
if args.verbose:
  print(log)
experiment_log.write(log)
experiment_log.close()
