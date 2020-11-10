import pickle
import torch
import numpy as np
from random import shuffle, seed
import matplotlib
import matplotlib.pyplot as plt
import argparse

from example_features import *

seed(0)

parser = argparse.ArgumentParser(description='Plot activation heat maps for encoder states.')
parser.add_argument('--states', dest="states",type=str,
                   help='List of states to plot e.g. 1,2,3,4.')
parser.add_argument('--representation', dest="representation", type=str,
                   help='Pickle file to read encoder representations from.')
args = parser.parse_args()

encoder_repr=args.representation
encoder_states=[int(s) for s in args.states.split(",")]

VALID_FILE="data/treebank-nouns.tsv.nom2gen.valid.annotated.csv"

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

data_hidden_states = pickle.load(open(encoder_repr,"br"))
annotated_data = [l.split(",") for l in open(VALID_FILE).read().split("\n")][1:]

def pad(maps):
    max_len = max([len(lemma) for lemma, _ in maps])
    maps = [(" " * (max_len - len(lemma)) + lemma,
             np.pad(ss,(max_len - len(lemma),0),mode='constant',constant_values=0)) for lemma, ss in maps]
    return maps

def get_activation_maps(data_hs,target_state):
    maps = []
    max_chars = []
    for datum, ss in data_hs:        
        lemma = datum[NOM].replace(" ","")
        ss = get_map(ss,target_state)
        maps.append([lemma,ss])
    maps = pad(maps)
    return maps

def plot(data_hs):
    shuffle(data_hs)
    data_hs = data_hs[:15]
    maps = get_activation_maps(data_hs,state)
    a = np.array([[x for x in ss.tolist()] for _,ss in maps])

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    for edge, spine in ax.spines.items():
        spine.set_visible(False)        
    im = ax.imshow(a,cmap="bwr",vmin=-1,vmax=1)
    for i, datum in enumerate(maps):
        for j, letter in enumerate(datum[0]):
            im.axes.text(j,i,letter,fontsize=12,ha='center',va='center')
    return plt

for state in encoder_states:
    for name, grad_test in zip("k p t qual quant".split(" "),
                               [is_k, is_p, is_t,qual_gradation, quant_gradation]):
        data_hs = [(d,hs) for d,hs in 
                   zip(annotated_data,data_hidden_states) if grad_test(d)]
        plt = plot(data_hs)
        plt.savefig("heatmaps/%s_%s.png" % (state,name))
    
    data_hs = [(d,hs) for d,hs in zip(annotated_data,data_hidden_states)
               if d[IS_GRADATION] == "yes"]
    plt = plot(data_hs)
    plt.savefig("heatmaps/%s_grad.png" % (state))

    data_hs = [(d,hs) for d,hs in zip(annotated_data,data_hidden_states)
               if d[IS_GRADATION] == "no"]
    plt = plot(data_hs)
    plt.savefig("heatmaps/%s_non_grad.png" % (state))
    plt.close()
