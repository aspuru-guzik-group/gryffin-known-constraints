#!/bin/bash

for d in $( ls -d ./*/ ); do
  echo -n "$d  --->  "
  cd $d
  python ../../check_num_repeats.py
  cd ../
done
