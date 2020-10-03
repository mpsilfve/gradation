import pickle
import torch
import numpy as np
from random import shuffle

HIDDEN_STATES="foo_step_3000.pt.treebank-nouns.tsv.nom2gen.valid.src.enc_states.pkl"
VALID_FILE="treebank-nouns.tsv.nom2gen.valid.annotated.csv"

# Fields in annotated valid data
NOM=0
GEN=1
IS_GRADATION=2
GRADATION_TYPE=3
AFFECTED_CONSONANT=4
DIRECTION=5

def get_map(t,state):
    t = t.squeeze(1)
    t = torch.cat([t,-t],dim=1)[:,state]
    return t.numpy()

data_hidden_states = pickle.load(open(HIDDEN_STATES,"br"))
annotated_data = [l.split(",") for l in open(VALID_FILE).read().split("\n")][1:]

def pad(maps):
    max_len = max([len(lemma) for lemma, _, gradation_index, _ in maps])
    maps = [(lemma,ch,gradation_index,np.pad(ss,(max_len - len(lemma)),mode='constant')) for lemma, ch, gradation_index, ss in maps]
    return maps

def get_activation_maps(ch,state):
    maps = []
    max_chars = []
    for datum, ss in zip(annotated_data, data_hidden_states):        
        if datum[AFFECTED_CONSONANT] == ch:
            lemma = datum[NOM].replace(" ","")
            gradation_index = lemma.rfind(ch)
            ss = get_map(ss,state)
            maps.append([lemma,ch,gradation_index,ss])
    maps = pad(maps)
    return maps

for state in [993,756]:
    for cons in "ktp":
        maps = get_activation_maps(cons,state)
        shuffle(maps)
        maps = maps[:15]
        a = np.array([ss for _,_,_,ss in maps])
        print(a)
        exit(1)

