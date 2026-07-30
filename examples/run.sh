#!/bin/bash


for is in "ls" "qdeim" "deim"
do
for i in 1 2 3
do
for N in 1024 2048 4096 8192 16384 32768 65536 131072 262144
do

#python3 poisson.py -n $N -log solver -is $is --seed $i
python3 poisson.py -n $N -log timing -is $is --seed $i --hardware neo

rm -rf /tmp/torchinductor_$(whoami)

done
done
done
