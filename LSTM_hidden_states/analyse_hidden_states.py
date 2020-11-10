import pickle
import torch
import numpy as np
import scipy
import scipy.stats
from scipy import stats
from sys import argv
import argparse

np.random.seed(12345678)

from example_features import *

VALID_FILE="data/treebank-nouns.tsv.nom2gen.valid.annotated.csv"

parser = argparse.ArgumentParser(description="Output table for difference in state activations for examples which undergo gradation and examples which don't.")
parser.add_argument('--representation', dest="representation", type=str,
                   help='Pickle file to read encoder representations from.')
parser.add_argument('--top_N', dest="top_N", type=int, default=5,
                   help='Compute activations for top N states.')
args = parser.parse_args()

encoder_repr=args.representation
top_n=args.top_N

def pool(t):
    t = t.squeeze(1)
    # Gradation occurs in the second to last character.
    return t[-2,:]

data_hidden_states = pickle.load(open(encoder_repr,"br"))
annotated_data = [l.split(",") for l in open(VALID_FILE).read().split("\n")][1:]

def get_deltas(top_n,condition1,condition2,data,states,target_states=None):
    """
    Compute the difference in activation for examples which satisfy
    condition1 vs. condition2 for each example and state in the
    tensor states. Return the mean difference and the activations for
    examples satifsfying both conditions.

    If target_states is supplied, return the difference in average activation for

    """
    deltas = []
    # Tere are 500 hidden states
    for v in range(500):
        all_val = []
        condition1_val = []
        condition2_val = []
        for datum, ss in zip(data, states):
            # Skip empty entry in the dataset
            if datum[NOM] == '_':
                continue

            s = pool(ss).numpy()
            all_val.append(s[v])
            if condition1(datum):
                condition1_val.append(s[v])
            # No gradation occurs in the form (exact criterion
            # determined by criterion2)
            elif condition2(datum):
                condition2_val.append(s[v])
        avg = sum(all_val)/len(all_val)
        yes_avg = sum(condition1_val)/len(condition1_val)
        no_avg = sum(condition2_val)/len(condition2_val)
        delta = yes_avg - no_avg
        deltas.append((np.abs(delta),v,condition1_val,condition2_val))
    if target_states:
        return [deltas[t] for t in target_states]
    else:
        deltas.sort(reverse=1)
        return [d[1] for d in deltas[:top_n]]

half_of_data = len(annotated_data)//2
all_set = get_deltas(top_n,is_gradation,cons_no_gradation,
                     annotated_data[half_of_data:],data_hidden_states[half_of_data:])
states = all_set

# Activations for different instances of gradation. K, P, T, all
# qualitative and all quantitative gradation.
k_set = get_deltas(top_n,is_k,cons_no_gradation,annotated_data[half_of_data:],
                   data_hidden_states[half_of_data:],target_states=states)
p_set = get_deltas(top_n,is_p,cons_no_gradation,annotated_data[half_of_data:],
                   data_hidden_states[half_of_data:],target_states=states)
t_set = get_deltas(top_n,is_t,cons_no_gradation,annotated_data[half_of_data:],
                   data_hidden_states[half_of_data:],target_states=states)
qual_set = get_deltas(top_n,qual_gradation,cons_no_gradation,annotated_data[half_of_data:],
                      data_hidden_states[half_of_data:],target_states=states)
quant_set = get_deltas(top_n,quant_gradation,cons_no_gradation,annotated_data[half_of_data:],
                       data_hidden_states[half_of_data:],target_states=states)

print("TOP %u states firing when gradation occurs:" % top_n)
print("\\begin{adjustbox}{width=0.32\\textwidth}")
print("\\begin{tabular}{lccccc}")
print("\\multicolumn{6}{c}{{\\sc Model %s}}\\\\" % argv[1])
print("\\toprule")
print("\\multirow{2}{*}{\\textbf{Gradation}} & \\multicolumn{5}{c}{\\textbf{State}}\\\\")
print(" & " + " & ".join([str(s) for s in states])+"\\\\")
print("\\midrule")
for cons, s in zip(["K","P","T","Qual.","Quant."],[k_set,p_set,t_set,qual_set,quant_set]):
    print(cons,end=" & ")
    for i in range(top_n):
        mean1 = sum([v for v in s[i][2]])/len(s[i][2])
        mean2 = sum([v for v in s[i][3]])/len(s[i][3])
        pval = scipy.stats.ttest_ind([v for v in s[i][2]],
                                     [v for v in s[i][3]], equal_var=False).pvalue
        if pval < 0.005:
            print("{\\bf%.3f}" % abs(mean1-mean2),end=" &" if i + 1 < top_n else " \\\\")
        else:
            print("%.3f" % (abs(mean1-mean2)),end=" &" if i + 1 < top_n else " \\\\")
    print()
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{adjustbox}")
