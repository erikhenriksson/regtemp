# Environment for running the scripts on LUMI. Source it from the repo in
# every session and in every batch job:
#
#     source setup_lumi.sh
#
# The virtual environment (venv/ in the repo) is loaded before the module.
# Scripts run from the repo; the data stay in $DATA, and trained models and
# results go to runs/ in the repo (not in git).

REPO=/scratch/project_462001491/ehenriks/regtemp
export DATA=/scratch/project_462001491/ehenriks/data/eng-1M

source $REPO/venv/bin/activate
module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=/scratch/project_462001491/ehenriks/hf_cache  # downloaded models
export HF_HUB_OFFLINE=1         # compute nodes: use the downloaded models only
export TORCH_COMPILE_DISABLE=1  # torch.compile crashes on LUMI's AMD GPUs
cd $REPO
