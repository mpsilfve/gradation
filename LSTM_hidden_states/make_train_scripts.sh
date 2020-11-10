for i in 1 2 3 4 5 6 7 8 9 10; do cat train.pbs | sed "s/MODEL/$i/g" > train.$i.pbs; done
