#!/bin/bash
#
#SBATCH -J 3d
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task 4
#SBATCH --time=200:00:00
#SBATCH --partition=cpu
#SBATCH --qos=nopreemption
#SBATCH --export=ALL
#SBATCH --output=gryffin.log
#SBATCH --gres=gpu:0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/h/matteoa/sw/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/h/matteoa/sw/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/h/matteoa/sw/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/h/matteoa/sw/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate feas

date >> gryffin.log
echo "" >> gryffin.log
python run.py
echo "" >> gryffin.log
date >> gryffin.log
