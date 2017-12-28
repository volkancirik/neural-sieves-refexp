#from __future__ import absolute_import
import argparse

def get_main_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--load',dest='dump', help='load a dataset dump',default='../data/stanford_cmn_refcocog_iou05_umd.NOtriplets.pkl')
  parser.add_argument('--cnn',dest='cnn',default='../data/stanford_cmn_refcocog_iou05.box+meta+smax.h5',help='cnn features filetype : h5')

  parser.add_argument('--hidden',dest='n_hidden', type=int, help='# of hidden units, default = 1000', default = 1000)
  parser.add_argument('--layers',dest='n_layer', type=int, help='# of layers for REL and LOC projections 1|2, default = 1', default = 1)
  parser.add_argument('--model',dest='model', help='model type cmn|cmn_loc, default:cmn_loc', default = 'cmn_loc')
  parser.add_argument('--dropout',dest='dropout', help='dropout rate, default=0', type=float, default = 0.)
  parser.add_argument('--clip',dest='clip', help='gradient clipping, default=10.0',type=float, default = 10.0)

  parser.add_argument('--timeout',dest='timeout',type=int, help='timeout in seconds, default = 300000', default = 300000)
  parser.add_argument('--seed',dest='seed',type=int, help='seed, default = 0',default=0)

  parser.add_argument('--epochs', type=int, default=6,help='# of epochs, default = 6')
  parser.add_argument('--val-freq', dest='val_freq', type=int, default=0,help='validate every n instances, 0 is for full pass over trn data, default = 0')
  parser.add_argument('--save-path', dest='save_path', type=str, default='exp',help='folder to save experiment')
  parser.add_argument('--resume', dest='resume', type=str, default='',help='resume from this model snapshot')
  parser.add_argument('--box-usage', dest='box_usage', type=int, default=0, help = "box features 0:cnn+spatial  1:cnn 2:spatial, default: 0")

  parser.add_argument('--verbose',dest='verbose', action = 'store_true', help='print to stdout')
  parser.add_argument('--use-outer',dest='use_outer', action = 'store_true', help='use outer product for features')
  parser.add_argument('--fusion',dest='fusion',help='fusion type mul|sum|concat default=mul',default='mul')
  parser.add_argument('--debug',dest='debug_mode', action = 'store_true', help='debug mode')
  parser.add_argument('--no-finetune',dest='finetune', action = 'store_false', help='do not finetune word embeddings')
  parser.add_argument('--optim',dest='optim',help='optimization method adam|sgd, default:sgd',default = 'sgd')
  parser.add_argument('--loss',dest='loss',help='loss for training nll|smargin|lamm|mbr, default:nll',default = 'nll')
  parser.add_argument('--lr',dest='lr',help='initial learning rate, default = 0.01',default = 0.01, type = float)
  parser.add_argument('--lr-min',dest='lr_min',help='minimum lr, default = 0.00001',default = 0.00001, type = float)
  parser.add_argument('--lr-decay',dest='lr_decay',help='learning rate decay, default = 0.4',default = 0.4, type = float)
  parser.add_argument('--w-decay',dest='weight_decay',help='weight decay, default = 0.0005',default = 0.0005, type = float)
  parser.add_argument('--encoder',dest='encoder',help='rnn encoder  lstm|gru, default:lstm',default = 'lstm')
  parser.add_argument('--phrase-context',dest='phrase_context', action = 'store_true', help='use phrase context for FLEX models')
  parser.add_argument('--only-spatial',dest='only_spatial', action = 'store_true', help='use only spatial features for REL() module')

  parser.add_argument('--frcnn',dest='frcnn',default='../data/projects/faster_rcnn_pytorch/models/VGGnet_fast_rcnn_iter_70000.h5', help='faster rcnn pytorch model path')
  parser.add_argument('--images',dest='images',default='./images', help='root folder of images')
  parser.add_argument('--top-n', dest='top_n',type=int, default=20,help='top-k of rpn proposals, default = 20')
  parser.add_argument('--out-file', dest='out_file', default='',help='output path for json file')
  parser.add_argument('--no-lang', dest='no_lang', action = 'store_true', help='do not use referring expression')
  parser.add_argument('--precision-k', dest='precision_k',type=int, default=5,help='precision @k, default = 5')
  parser.add_argument('--yo',dest='yo', action = 'store_true', help='send logs via yo!')
  args = parser.parse_args()
  return args

def get_cmn_imbd():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed',dest='seed',type=int, help='seed, default = 1',default=1)
  parser.add_argument('--save',dest='dump', help='dump the dataset')
  parser.add_argument('--data-root',dest='path',default='/home/vcirik/projects/cmn/preprocess/REFCOCOG/',
                      help='path to data folder containing cmn_imbd')
  parser.add_argument('--task',dest='task',default='refcocog',
                      help='task name refcocog|visual7w, default: refcocog')
  parser.add_argument('--blacklist',dest='blacklist',default='../data/blacklist.txt', help="the list of blacklisted words, default=../data/blacklist.txt")
  parser.add_argument('--mapping',dest='mapping',default='../data/mapping.txt', help="mapping of typos, default=../data/mapping.txt")
  parser.add_argument('--wordvec',dest='wordvec',default='../data/wordvec.glove',help="word vectors, default=../data/wordvec.glove")
  parser.add_argument('--val-prob',dest='val_prob',help='probability of trn instance become val instance, default = 0.025',default = 0.025, type = float)
  parser.add_argument('--use-triplets',dest='use_triplets', action = 'store_true', help='use collapsed triplets')
  parser.add_argument('--triplet-mode', dest='mode', type=int, default=1,help='triplet mode 0=depth 1, 1= depth n, 2= depth n with candidate fixes, default = 1')
  parser.add_argument('--gold-iou',dest='gold_iou', type = float, default = 0.5, help='0.5 < = gold label iou <= 1.0, default = 0.5')
  parser.add_argument('--tree-type',dest='tree_type',help="tree type berkeley|standord default=stanford",default='stanford')
  parser.add_argument('--ary',dest='ary',help="binary or nary tree .bp|.np default=.np",default='.np')
  parser.add_argument('--split',dest='split_map',help="split map pickle file ../data/umd_split_map.pkl|'' default=''",default='')
  parser.add_argument('--perturb',dest='perturb',help="perturbation function shuffle|shuffle_pos|noun|adj|noun+adj|adj+noun|headp default='shuffle'",default='shuffle')
  parser.add_argument('--imdbs',dest='imdbs',help="imdbs for headp perturbation '' or '/home/vcirik/projects/cmn/exp-refgoog/data/imdb/imdb_trn.npy /home/vcirik/projects/cmn/exp-refgoog/data/imdb/imdb_val.npy' default=''",default='')

  args = parser.parse_args()
  return args

def get_baseline_args():

  parser = argparse.ArgumentParser()

  parser.add_argument('--load',dest='dump', help='load a dataset dump',default='../data/stanford_cmn_refcocog_iou05_umd.NOtriplets.pkl')
  parser.add_argument('--cnn',dest='cnn',default='../data/stanford_cmn_refcocog_iou05.box+meta+smax.h5',help='cnn features filetype : h5')
  parser.add_argument('--hidden',dest='n_hidden', type=int, help='# of hidden units, default = 1000', default = 1000)
  parser.add_argument('--layers',dest='n_layer', type=int, help='# of layers for REL and LOC projections 1|2, default = 1', default = 1)
  parser.add_argument('--model',dest='model', help='model type cmn|cmn_loc, default:cmn_loc', default = 'cmn_loc')
  parser.add_argument('--dropout',dest='dropout', help='dropout rate, default=0', type=float, default = 0.)
  parser.add_argument('--clip',dest='clip', help='gradient clipping, default=10.0',type=float, default = 10.0)

  parser.add_argument('--timeout',dest='timeout',type=int, help='timeout in seconds, default = 300000', default = 300000)
  parser.add_argument('--seed',dest='seed',type=int, help='seed, default = 0',default=0)

  parser.add_argument('--epochs', type=int, default=6,help='# of epochs, default = 6')
  parser.add_argument('--val-freq', dest='val_freq', type=int, default=0,help='validate every n instances, 0 is for full pass over trn data, default = 0')
  parser.add_argument('--save-path', dest='save_path', type=str, default='exp',help='folder to save experiment')
  parser.add_argument('--resume', dest='resume', type=str, default='',help='resume from this model snapshot')
  parser.add_argument('--box-usage', dest='box_usage', type=int, default=0, help = "box features 0:cnn+spatial  1:cnn 2:spatial, default: 0")

  parser.add_argument('--verbose',dest='verbose', action = 'store_true', help='print to stdout')
  parser.add_argument('--use-outer',dest='use_outer', action = 'store_true', help='use outer product for features')
  parser.add_argument('--fusion',dest='fusion',help='fusion type mul|sum|concat default=mul',default='mul')
  parser.add_argument('--debug',dest='debug_mode', action = 'store_true', help='debug mode')
  parser.add_argument('--no-finetune',dest='finetune', action = 'store_false', help='do not finetune word embeddings')
  parser.add_argument('--optim',dest='optim',help='optimization method adam|sgd, default:sgd',default = 'sgd')
  parser.add_argument('--loss',dest='loss',help='loss for training nll|smargin|lamm|mbr, default:smargin',default = 'smargin')
  parser.add_argument('--lr',dest='lr',help='initial learning rate, default = 0.01',default = 0.01, type = float)
  parser.add_argument('--lr-min',dest='lr_min',help='minimum lr, default = 0.00001',default = 0.00001, type = float)
  parser.add_argument('--lr-decay',dest='lr_decay',help='learning rate decay, default = 0.4',default = 0.4, type = float)
  parser.add_argument('--w-decay',dest='weight_decay',help='weight decay, default = 0.0005',default = 0.0005, type = float)
  parser.add_argument('--encoder',dest='encoder',help='rnn encoder  lstm|gru, default:lstm',default = 'lstm')
  parser.add_argument('--phrase-context',dest='phrase_context', action = 'store_true', help='use phrase context for FLEX models')
  parser.add_argument('--only-spatial',dest='only_spatial', action = 'store_true', help='use only spatial features for REL() module')

  parser.add_argument('--frcnn',dest='frcnn',default='../data/projects/faster_rcnn_pytorch/models/VGGnet_fast_rcnn_iter_70000.h5', help='faster rcnn pytorch model path')
  parser.add_argument('--images',dest='images',default='./images', help='root folder of images')
  parser.add_argument('--top-n', dest='top_n',type=int, default=20,help='top-k of rpn proposals, default = 20')
  parser.add_argument('--out-file', dest='out_file', default='',help='output path for json file')
  parser.add_argument('--no-lang', dest='no_lang', action = 'store_true', help='do not use referring expression')
  parser.add_argument('--precision_f1', dest='precision_f1', type = float, default=3.0,help='precision for filter 1, default = 3')
  parser.add_argument('--precision_f2', dest='precision_f2', type = float, default=3.0,help='precision for filter 2, default = 3')

  parser.add_argument('--refexp-root', dest='refexp_root', default='../data',help='root for google_refexp_{train|val}_201511_coco_aligned_mcg_umd.json, default = ../data')

  parser.add_argument('--coco-root', dest='coco_root', default='../data',help='root for instances_{train|val}2014.json, default = ../data')

  parser.add_argument('--filter-model',dest = 'filter_model', help="filtering model", default='exp-NOLANGUMD0/snapshot.MODELcmn_loc.N_HIDDEN1000.N_LAYER1.DROPOUT0.0.USE_OUTERFalse.BOX_USAGE0.LOSSnll.OPTIMsgd.LR0.01.LR_MIN1e-05.LR_DECAY0.4.WEIGHT_DECAY0.0005.CLIP10.0.ENCODERlstm.ONLY_SPATIALFalse.PHRASE_CONTEXTFalse.DEBUGFalse.WORD_DIM300.model.best')

  parser.add_argument('--baseline-mode', dest='baseline_mode', help='baseline mode oracle|vanilla default=vanilla', default = 'vanilla')

  parser.add_argument('--order-feats',dest='order_feats', action = 'store_true', help='use order features for boxes')
  parser.add_argument('--box-filter-model',dest = 'box_filter_model', help="box type filtering model", default = '')
  parser.add_argument('--threshold',dest='threshold',help='threshold for box filter prediction, default = 0.5',default = 0.5, type = float)
  parser.add_argument('--yo',dest='yo', action = 'store_true', help='send logs via yo!')
  args = parser.parse_args()
  return args
