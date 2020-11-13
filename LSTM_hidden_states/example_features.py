# Fields in annotated valid data
NOM=0 
GEN=1
IS_GRADATION=2
GRADATION_TYPE=3
AFFECTED_CONSONANT=4
DIRECTION=5

def is_k(datum):
    return datum[IS_GRADATION] == "yes" and datum[NOM].split(" ")[-2] == "k"

def is_p(datum):
    return datum[IS_GRADATION] == "yes" and datum[NOM].split(" ")[-2] == "p"

def is_t(datum):
    return datum[IS_GRADATION] == "yes" and datum[NOM].split(" ")[-2] == "t"

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
