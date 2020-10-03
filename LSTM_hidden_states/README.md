To run inference on the model (outputs a file pred.txt):

python3 ../OpenNMT-py/translate.py --src treebank-nouns.tsv.nom2gen.valid.src --model foo_step_3000.pt

To find states which fire strongly when the stem undergoes gradation for k, p or t, run:

python3 analyse_hidden_states.py

To track the activation of the a hidden state in different positions the input strings, run:

python3 find_state_max.py