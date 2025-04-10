#!/bin/bash
#SBATCH --job-name=bFOF_z0              # Job name (keep to <= 8 characters)
#SBATCH --account=torrey-group          # Account to charge
#SBATCH --partition=standard            # Partition to run on
#SBATCH --ntasks=1                      # Run on a single CPU
#SBATCH --mem=5gb                       # Job memory request
#SBATCH --time=08:00:00                 # Time limit hrs:min:sec
#SBATCH --output=out_%j.log             # Standard output and error log
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yja6qa@virginia.edu # Where to send mail
pwd; hostname; date

module purge
module load anaconda
# conda activate py3 ## Your python environment

cd /home/yja6qa/arepo_package/ ## (optional) change directories to where the script is

python M_Sigma.py ## Your example script

date
