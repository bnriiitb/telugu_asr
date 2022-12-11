#!/bin/bash
#SBATCH --account=dgx2
#SBATCH --job-name=job_test          # Job name
#SBATCH --ntasks=1                   # Number of MPI tasks (i.e. processes) (Restricted to 20 tasks per user a/c)
#SBATCH --cpus-per-task=1            # Number of cores per MPI task 
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --gres=gpu:1                 # Maximum number of GPUs to be allocated (Restricted to 1 GPU per user a/c)
#SBATCH --ntasks-per-node=1          # Maximum number of tasks on each node (Restricted to 20 tasks per user a/c)
#SBATCH --output=mpi_test_%j.log     # Path to the standard output and error files relative to the working directory

# The normal method to kill a Slurm job is:
#     $ scancel <jobid>

# You can find your jobid with the following command:
#     $ squeue -u $USER

# If the the job id is 1234567 then to kill the job:
#     $ scancel 1234567


# sbatch script.sh --ntasks=20 --cpus-per-task=16 --nodes=20 --ntasks-per-node=20

source /raid/cs20mds14030/miniconda3/etc/profile.d/conda.sh
# cd telugu_asr
# conda create --name telugu_asr python=3.8 --yes
conda activate telugu_asr
# conda install pytorch torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia --yes
# cd telugu_asr
# pip3 install -r requirements.txt
cd notebooks
# python train.py
python validate.py
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
mpirun -np 1 ./a.out >> output.txt
