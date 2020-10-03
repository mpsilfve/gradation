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

def maxpool(t):
    t = t.squeeze(1)
    t = torch.cat([t,-t],dim=1)
    return t.max(dim=0)[0]


data_hidden_states = pickle.load(open(HIDDEN_STATES,"br"))
annotated_data = [l.split(",") for l in open(VALID_FILE).read().split("\n")][1:]

def get_deltas(ch,top_n):
    deltas = []
    for v in range(1000):
        all_val = []
        yes_val = []
        no_val = []
        for datum, ss in zip(annotated_data, data_hidden_states):
            s = maxpool(ss).numpy()
            all_val.append(s[v])
            if datum[AFFECTED_CONSONANT] == ch:
                yes_val.append(s[v])
            # No gradation occurs in the form
            elif datum[IS_GRADATION] == "no":
                no_val.append(s[v])
        avg = sum(all_val)/len(all_val)
        yes_avg = sum(yes_val)/len(yes_val)
        no_avg = sum(no_val)/len(no_val)
        delta = abs(yes_avg - no_avg)
        deltas.append((delta,v))
    deltas.sort()
    return deltas[-top_n:]

k_set = get_deltas("k",30)
t_set = get_deltas("t",30)
p_set = get_deltas("p",30)

print("TOP 30 states firing when gradation occurs for each consonant:")
print("K",k_set)
print("\nP",p_set)
print("\nT",t_set)

k_set = set([x[1] for x in k_set])
p_set = set([x[1] for x in p_set])
t_set = set([x[1] for x in t_set])

print("States firing when gradation occurs for two distinct consonants:")
print("K isect P",k_set.intersection(p_set))
print("K isect T",k_set.intersection(t_set))
print("P isect T",p_set.intersection(t_set))
