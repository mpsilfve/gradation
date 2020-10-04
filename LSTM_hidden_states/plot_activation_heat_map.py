import pickle
import torch
import numpy as np
from random import shuffle, seed
import matplotlib
import matplotlib.pyplot as plt

seed(0)

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

for state in [476,410,256,349,446]:
    for cons in "kpt":
        data_hs = [(d,hs) for d,hs in zip(annotated_data,data_hidden_states)
                   if d[AFFECTED_CONSONANT] == cons]
        plt = plot(data_hs)
        plt.savefig("%s_%s.png" % (state,cons))
    
    data_hs = [(d,hs) for d,hs in zip(annotated_data,data_hidden_states)
               if d[IS_GRADATION] == "yes"]
    plt = plot(data_hs)
    plt.savefig("%s_grad.png" % (state))

    data_hs = [(d,hs) for d,hs in zip(annotated_data,data_hidden_states)
               if d[IS_GRADATION] == "yes"]
    plt = plot(data_hs)
    plt.savefig("%s_non_grad.png" % (state))
    plt.close()
