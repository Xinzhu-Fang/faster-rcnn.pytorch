#! /bin/bash
#

#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:tesla-k80:1

my_command="python demo.py --net vgg16 --dataset pascal_voc\
                   --checksession 1 --checkepoch 1 --checkpoint 10021 \
                   --cuda --load_dir models > my_demo.txt"
echo $my_command
eval $my_command

