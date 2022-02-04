#!/bin/bash

for d in $( ls -d ./*/ ); do
  echo $d
  cd $d
  sbatch submit.sh
  cd ../
done
