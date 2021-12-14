#!/bin/bash
#SBATCH --gres=gpu:2         # Number of GPUs (per node)
#SBATCH --cpus-per-task=4    # Number of CPUs
#SBATCH --mem=85G            # memory (per node)
#SBATCH --time=0-12:00       # time (DD-HH:MM)
#SBATCH --partition=main     # priority: unkillable > main > long
#SBATCH --job-name=KABROLG   #

module load cuda/10.1
source ../lm/bin/activate

filename=train.sh

chmod +x $filename
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

. $filename

"""
############## README : Before runing this file on the cluster #################

module load python/3.7
virtualenv lm
source lm/bin/activate
pip install --upgrade pip

git clone https://github.com/Tikquuss/lm
cd lm
pip install -r requirements.txt
### for `import pytorch_lightning as pl` issues
pip3 install packaging
pip install importlib-metadata
pip install transformers -U
### for `from language_modelling import LMLightningModule` issues
pip3 install python-dateutil
pip uninstall attr
pip install attrs

tmux

chmod +x cluster.sh

#
salloc --gres=gpu:2 -c 4 --mem=32Gb --time=12:00:00 --partition=main --job-name=KABROLG
. cluster.sh
# or 
srun --gres=gpu:2 -c 4 --mem=32Gb --time=12:00:00 --partition=main --job-name=KABROLG . cluster.sh
# or (see SBATCH parameters at the beginning of the file)
sbatch . cluster.sh
#
"""