from sys import argv
import argparse

from example_features import *

def get_args():
    parser = argparse.ArgumentParser(description='Plot activation heat maps for encoder states.')
    parser.add_argument('--sys', dest="sys",type=str,
                        help='System output file',
                        required=True)
    parser.add_argument('--gold', dest="gold", type=str,
                        help='Gold standard outputs',
                        required=True)
    parser.add_argument('--annotated', dest="annotated", type=str,
                        help='A CSV file containing annotations for the input.',
                        default=None)
    return parser.parse_args()

def eval(sys_fn, gold_fn,annotations=None, test=None):
    corr = 0
    total = 0
    for i, (l1, l2) in enumerate(zip(open(sys_fn),open(gold_fn))):
        d = None
        if annotations:
            d = annotations[i].split(",")
        l1 = l1.strip()
        l2 = l2.strip()
        if test:
            if test(d):
                corr += (l1 == l2)
                total += 1
        else:
            corr += (l1 == l2)
            total += 1
    return corr, total, 100*corr/total

if __name__=="__main__":
    args = get_args()

    print("Correct forms: %u, Total forms: %u, Accuracy for all forms: %.2f" %
          eval(args.sys,args.gold))

    if args.annotated:
        # Read annotations. Skip the header line.
        annotations = open(args.annotated).read().split("\n")[1:]

        print("Correct forms: %u, Total forms: %u, Accuracy for all grad forms: %.2f" %
              eval(args.sys,args.gold,annotations,is_gradation))
        print("Correct forms: %u, Total forms: %u, Accuracy for all non-grad forms: %.2f" %
              eval(args.sys,args.gold,annotations,no_gradation))
