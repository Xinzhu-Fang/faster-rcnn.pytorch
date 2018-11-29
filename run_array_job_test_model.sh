#! /bin/bash
#

#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:tesla-k80:1

my_command="python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 1 --checkpoint 10021 \
                   --cuda > my_test.txt"
echo $my_command
eval $my_command

