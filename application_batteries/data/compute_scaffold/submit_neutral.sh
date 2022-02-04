#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=3:00:00
##SBATCH --job-name neutral-scaffold
##SBATCH --output=neutral-scaffold_%j.out
#SBATCH -o job_%j.log

# command line arg gives the path to the .com files
args=("$@")

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

#g16
export g16root="/project/a/aspuru/opt/gaussian"
gr=$g16root
export GAUSS_EXEDIR="$gr/g16C01/bsd:$gr/g16C01"
export GAUSS_LEXEDIR="$gr/g16C01/linda-exe"
export GAUSS_ARCHDIR="$gr/g16C01/arch"
export GAUSS_BSDDIR="$gr/g16C01/bsd"
export LD_LIBRARY_PATH="$GAUSS_EXEDIR:$LD_LIBRARY_PATH"
export PATH="$PATH:$gr/gauopen:$GAUSS_EXEDIR"
GAUSS_SCRDIR=$SCRATCH

#Running the g16 jobs
g16 <$SLURM_SUBMIT_DIR/${args[0]}/n.com>  $SLURM_SUBMIT_DIR/${args[0]}/n.log
g16 <$SLURM_SUBMIT_DIR/${args[0]}/c.com>  $SLURM_SUBMIT_DIR/${args[0]}/c.log
