#!/bin/bash
#SBATCH --job-name=iwp_batch_inference_notebook
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --account=bbou-delta-cpu
#SBATCH --time=01:00:00

#SBATCH --ntasks=1
##SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem-per-cpu=8g       
#SBATCH --output=%N-%j.out


conda activate /projects/bbou/mohanar2/envs/ray_sandbox

export HEAD_NODE=$(hostname) # store head node's address
export RAY_PORT=34567

# Launch single-node ray cluster as a background process: only node is the head node! 
ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=$SLURM_CPUS_PER_TASK  --block &
sleep 30

srun jupyter-notebook --no-browser --ip=0.0.0.0