for i in 1 2 3 4 5 6 7 8 9 10; do cat config.yaml | sed "s/MODEL/$i/" > config.$i.yaml; done
