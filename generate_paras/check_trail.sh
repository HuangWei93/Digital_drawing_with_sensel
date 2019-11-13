#!/bin/bash
for i in $(seq "$1")
do
 echo "checking collected trails $i"
 python plot_trail.py $i
done
