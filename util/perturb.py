#from __future__ import absolute_import
import nltk
from random import shuffle, seed
from collections import defaultdict

def perturb(tokens, fn):
  if fn == 'shuffle':
    shuffle(tokens)
    return tokens

  if fn == 'shuffle_pos':
      tagmap = {'NN' : 'noun', 'NNS' : 'noun', 'NNP' : 'noun', 'NNPS' : 'noun', 'JJ' : 'adj', 'JJS' : 'adj', 'JJR' : 'adj'}
      tagset = set(['NN','NNS','NNP','NNPS','JJ','JJS','JJR','noun','adj'])
      tags = nltk.pos_tag(tokens)
      reference = {}
      tag_list = defaultdict(list)
      must_shuffle = False
      for i,(w,tag) in enumerate(tags):
        tag = tagmap.get(tag,tag)
        tag_list[tag].append(i)
        reference[i] = len(tag_list[tag])-1
      for tag in tagset:
        vocab = set([tokens[idx] for idx in tag_list[tag]])
        if len(tag_list[tag]) >= 2 and len(vocab)>=2:
          must_shuffle = True

      while True:
        shuffled = ['']*len(tags)
        for tag in tag_list:
          tag = tagmap.get(tag,tag)
          if tag in tagset:
            shuffle(tag_list[tag])

        for i in xrange(len(tokens)):
          tag = tags[i][1]
          tag = tagmap.get(tag,tag)
          shuffled[i] = tokens[tag_list[tag][reference[i]]]
        if " ".join(tokens) == " ".join(shuffled) and must_shuffle:
          continue
        break
      return shuffled

  if fn in set(["noun","adj","noun+adj","adj+noun"]):
    tags = nltk.pos_tag(tokens)
    new_tokens = []
    for (w,tag) in tags:
      if (tag == 'NN' or tag == "NNS" or tag == "NNP" or tag == "NNPS") and 'noun' in fn:
        new_tokens += [w]
      if (tag == 'JJ' or tag == "JJS" or tag == "JJR") and 'adj' in fn:
        new_tokens += [w]
    if len(new_tokens) == 0:
      new_tokens = tokens
    return new_tokens

  if fn == 'flip':
    tagmap = {'NN' : 'noun', 'NNS' : 'noun', 'NNP' : 'noun', 'NNPS' : 'noun'}
    tagset = set(['NN','NNS','NNP','NNPS','noun'])

    tags = nltk.pos_tag(tokens)
    reference = {}
    tag_list = defaultdict(list)
    must_shuffle = False
    for i,(w,tag) in enumerate(tags):
      tag = tagmap.get(tag,tag)
      tag_list[tag].append(i)
      reference[i] = len(tag_list[tag])-1
    for tag in tagset:
      vocab = set([tokens[idx] for idx in tag_list[tag]])
      if len(tag_list['noun']) >= 2 and len(vocab)>=2:
        must_shuffle = True

    if len(tag_list['noun']) < 1:
      target = None
      must_shuffle = False
    else:
      target = tag_list['noun'][0]

    while True:
      shuffle(tag_list['noun'])
      if must_shuffle and tag_list['noun'][0] == target:
        continue
      break

    shuffled = ['']*len(tags)
    for i in xrange(len(tokens)):
      tag = tags[i][1]
      tag = tagmap.get(tag,tag)
      shuffled[i] = tokens[tag_list[tag][reference[i]]]
    return shuffled
  if fn=='shuffle_pos_target':
    tagmap = {'NN' : 'noun', 'NNS' : 'noun', 'NNP' : 'noun', 'NNPS' : 'noun', 'JJ' : 'adj', 'JJS' : 'adj', 'JJR' : 'adj'}
    tagset = set(['NN','NNS','NNP','NNPS','JJ','JJS','JJR','noun','adj'])

    tags = nltk.pos_tag(tokens)
    reference = {}
    tag_list = defaultdict(list)
    must_shuffle = False

    for i,(w,tag) in enumerate(tags):
      tag = tagmap.get(tag,tag)
      tag_list[tag].append(i)
      reference[i] = len(tag_list[tag])-1
    for tag in tagset:
      vocab = set([tokens[idx] for idx in tag_list[tag]])
      if len(tag_list['noun']) >= 2 and len(vocab)>=2:
        must_shuffle = True

    if len(tag_list['noun']) < 1:
      target = None
      must_shuffle = False
    else:
      target = tag_list['noun'][0]

    while True:
      shuffled = ['']*len(tags)
      for tag in tag_list:
        tag = tagmap.get(tag,tag)
        if tag in tagset:
          shuffle(tag_list[tag])

      for i in xrange(len(tokens)):
        tag = tags[i][1]
        tag = tagmap.get(tag,tag)
        shuffled[i] = tokens[tag_list[tag][reference[i]]]
      if must_shuffle and (" ".join(tokens) == " ".join(shuffled) or tag_list['noun'][0] == target):
        continue
      break
    return shuffled
