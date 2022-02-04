#! /bin/bash

for optimizer in 'adam' 'genetic'; do
	folder="opt_${optimizer}"
	script="$folder/run.py"
	if [ ! -d $folder ]; then
		echo "Creating  ${folder}"
		mkdir $folder
		sed "s/_OPTIMIZER_/${optimizer}/g" "run_template.py" > $script
		cp submit.sh "$folder/"
	fi
done
