# Environment for running the scripts on LUMI. Source it in every session
# and in every batch job:
#
#     source setup_lumi.sh
#
# Copy it to the data folder and adjust the paths and the module lines to
# your own setup; the virtual environment is loaded before the module.

source /scratch/project_462001491/ehenriks/temporaltest/v/bin/activate
module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=/scratch/project_462001491/ehenriks/hf_cache  # downloaded models
export HF_HUB_OFFLINE=1         # compute nodes: use the downloaded models only
export TORCH_COMPILE_DISABLE=1  # torch.compile crashes on LUMI's AMD GPUs
cd /scratch/project_462001491/ehenriks/data/eng-1M
