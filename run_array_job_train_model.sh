#! /bin/bash
#

#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:tesla-k80:1

my_command="python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --epochs 1                  --cuda > train_net_res101_vg.txt"
echo $my_command
eval $my_command
