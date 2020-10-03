import pickle
import torch

HIDDEN_STATES="foo_step_3000.pt.treebank-nouns.tsv.nom2gen.valid.src.enc_states.pkl"
VALID_FILE="treebank-nouns.tsv.nom2gen.valid.annotated.csv"

# Fields in annotated valid data
NOM=0
GEN=1
IS_GRADATION=2
GRADATION_TYPE=3
AFFECTED_CONSONANT=4
DIRECTION=5

def max_index(t,state):    
    t = t.squeeze(1)
    t = torch.cat([t,-t],dim=1)[:,state]
    return t.max(dim=0)[1]

data_hidden_states = pickle.load(HIDDEN_STATES,"br"))
annotated_data = [l.split(",") for l in open(VALID_FILE).read().split("\n")][1:]

def get_max_index(ch,state):
    max_indices = []
    max_chars = []
    for datum, ss in zip(annotated_data, data_hidden_states):        
        len_s = ss.size()[0]
        s = max_index(ss,state).numpy()
        if datum[AFFECTED_CONSONANT] == ch:
            max_indices.append(len_s - s)
            wf = datum[0].split(" ")
            wf[s] = wf[s].upper()
            max_chars.append("".join(wf))

    return sum(max_indices)/len(max_indices), max_chars

for state in [993,756]:
    for cons in "ktp":
        print(state,cons,get_max_index(cons,state)[1])


