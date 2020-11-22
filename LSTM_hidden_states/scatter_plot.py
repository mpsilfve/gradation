import pickle
import matplotlib.pyplot as plt
from random import random

NOM=0 
GEN=1
IS_GRADATION=2
GRADATION_TYPE=3
AFFECTED_CONSONANT=4
DIRECTION=5

HIDDEN_STATES="models/gradation_3_step_3000.pt.treebank-nouns.tsv.nom2gen.valid.src.enc_states.pkl"
VALID_FILE="data/treebank-nouns.tsv.nom2gen.valid.annotated.csv"

data_hidden_states = pickle.load(open(HIDDEN_STATES,"br"))
annotated_data = [l.split(",") for l in open(VALID_FILE).read().split("\n")][1:]

def is_k(datum):
    return datum[IS_GRADATION] == "yes" and datum[AFFECTED_CONSONANT] == "k"

def is_p(datum):
    return datum[IS_GRADATION] == "yes" and datum[AFFECTED_CONSONANT] == "p"

def is_t(datum):
    return datum[IS_GRADATION] == "yes" and datum[AFFECTED_CONSONANT] == "t"

def no_gradation(datum):
    return datum[IS_GRADATION] == "no" 

def is_gradation(datum):
    return datum[IS_GRADATION] == "yes" 

def qual_gradation(datum):
    return datum[GRADATION_TYPE] == "qualitative" 

def quant_gradation(datum):
    return datum[GRADATION_TYPE] == "quantitative" 

def cons_no_gradation(datum):
    return datum[NOM].split(" ")[-2] in "bqwrtpsdfghjklzxcvbnm" and datum[IS_GRADATION] == "no"

def is_inverse(datum):
    return datum[-1] == "inverse"

def pool(t):
    t = t.squeeze(1)#.abs()
#    t = torch.cat([t,-t],dim=1)
# Gradation occurs in the second to last character.
    return t[-2,:]

from sys import argv

if __name__=="__main__":
    s1 = int(argv[1])
    s2 = int(argv[2])

    k_grad_act1 = []
    k_grad_act2 = []
    t_grad_act1 = []
    t_grad_act2 = []
    p_grad_act1 = []
    p_grad_act2 = []
    inv_k_grad_act1 = []
    inv_k_grad_act2 = []
    inv_t_grad_act1 = []
    inv_t_grad_act2 = []
    inv_p_grad_act1 = []
    inv_p_grad_act2 = []
    no_grad_act1 = []
    no_grad_act2 = []

    for datum, ss in zip(annotated_data, data_hidden_states):
        try:
            act1 = pool(ss).numpy()[s1] + random()*0.05
            act2 = pool(ss).numpy()[s2] + random()*0.05
            if is_k(datum):
                if is_inverse(datum):
                    inv_k_grad_act1.append(act1)
                    inv_k_grad_act2.append(act2)
                else:
                    k_grad_act1.append(act1)
                    k_grad_act2.append(act2)

            elif is_p(datum):
                if is_inverse(datum):
                    inv_p_grad_act1.append(act1)
                    inv_p_grad_act2.append(act2)
                else:
                    p_grad_act1.append(act1)
                    p_grad_act2.append(act2)

            elif is_t(datum):
                if is_inverse(datum):
                    inv_t_grad_act1.append(act1)
                    inv_t_grad_act2.append(act2)
                    print(datum,act1,act2)
                else:
                    t_grad_act1.append(act1)
                    t_grad_act2.append(act2)

            else:
                if act1 < -0.4:
                    print(datum[0].replace(" ",""),datum[1].replace(" ",""))
                no_grad_act1.append(act1)
                no_grad_act2.append(act2)
        except IndexError:
            print("Skipping",datum)
            continue
    plt.scatter(no_grad_act1, no_grad_act2, marker="$\cdot$", c="black", edgecolors='none',
                label="no gradation")
    plt.scatter(k_grad_act1, k_grad_act2, marker="$k$", c="blue", edgecolors='none',s=80,                
                label="direct k-gradation")
    plt.scatter(p_grad_act1, p_grad_act2, marker="$p$", c="orange", edgecolors='none', s=80,
                label="direct p-gradation")
    plt.scatter(t_grad_act1, t_grad_act2, marker="$t$", c="green", edgecolors='none', s=80,
                label="direct t-gradation")
    plt.scatter(inv_k_grad_act1, inv_k_grad_act2, marker="$K$", c="blue", edgecolors='none', s=80,
                label="indirect k-gradation")
    plt.scatter(inv_p_grad_act1, inv_p_grad_act2, marker="$P$", c="orange", edgecolors='none', s=80,
                label="indirect p-gradation")
    plt.scatter(inv_t_grad_act1, inv_t_grad_act2, marker="$T$", c="green", edgecolors='none',s=80,
                label="indirect t-gradation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,numpoints=1)
    plt.savefig("activation_plot.pdf",bbox_inches='tight')
