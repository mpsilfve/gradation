STATES=338,166,136,3,70,340,100,29,380,0,491,51,50,440,453,438,197,391,493,259

for f in 1 0 -1 -2 -3 -4 -5 -6 ; do echo $f && python3 ../OpenNMT-py/translate.py --model gradation_step_3000.pt --src gradating_src.txt --tgt gradating_trg.txt -perturb_states $STATES -scaling_factor $f --batch_size 1 && echo "No gradation:" && python3 compare.py pred.txt gradating_trg.txt && echo "Gradation" && python3 compare.py pred.txt gradations.trg ; done
