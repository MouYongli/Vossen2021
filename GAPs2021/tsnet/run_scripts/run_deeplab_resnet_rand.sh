#!/usr/local_rwth/bin/zsh
### ask for 10 GB memory
#SBATCH --mem-per-cpu=20G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=gaps
### job run time
#SBATCH --time=20:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=deeplab_resnet_rand.%J.txt
###
#SBATCH --mail-type=ALL
###
#SBATCH --mail-user=yongli.mou@rwth-aachen.de
### request a GPU
#SBATCH --gres=gpu:pascal:1

### begin of executable commands
cd $HOME/GAPS2021/pytorch-deeplab-xception-master
### load modules
module switch intel gcc
module load python/3.6.8
module load cuda/102
module load cudnn/8.0.5

python3 train.py --model deeplab --backbone resnet --crop-strategy rand
