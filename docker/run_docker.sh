#!/bin/bash

if [ -z $1$ ]
  then
    container="ahsanmah/mri-tf1:latest"
else
  container=$1
fi

# -u 298493:1001 \ -u $(id -u):$(id -g) 
# --group-add users
# -v "/$(pwd)/notebooks":"/.jupyter/" \
# -u $(id -u):$(id -g) \
# --group-add users \
# -e JUPYTER_TOKEN="niral" \
# -e PASSWORD=niral \s

docker run \
	--rm \
	-d \
	-it \
	--name ahsan-tfv2.4 \
	-p 9000:8888 \
	-p 9001-9003:9001-9003 \
	--runtime=nvidia \
	--gpus=all \
	--entrypoint="" \
	--mount type=bind,src=/ASD/ahsan_projects/,target=/home \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
	--mount type=bind,src="/Human/conte_projects/CONTE_NEO/Data/",target=/NEO \
	--mount type=bind,src="/Human2/CONTE2/Data/",target=/CONTE2 \
	ahsanmah/msma:latest \
	bash -c 'source /etc/bash.bashrc &&
	# export PATH="$PATH:/.local/bin"
	# export PATH="$PATH:/home/user/.local/bin"
	echo Starting Jupyter Lab...
	jupyter lab --notebook-dir=/ --ip 0.0.0.0 --no-browser --allow-root'

#  jupyter lab --notebook-dir=/ --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='''
# 	--mount type=bind,src=/ASD/ahsan_projects/.vscode-server/,target=/root/.vscode-server/ \
