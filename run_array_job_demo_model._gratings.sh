#! /bin/bash
#

#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:tesla-k80:1

my_command1="python Cindy_demo_gratings.py --net res101 --dataset pascal_voc\
                   --checksession 1 --checkepoch 7 --checkpoint 10021 \
                   --cuda --stimuli gratings"
echo $my_command1
eval $my_command1

my_command2="python Cindy_demo_gratings.py --net vgg16 --dataset pascal_voc\
                   --checksession 1 --checkepoch 1 --checkpoint 10021 \
                   --cuda --stimuli gratings"
echo $my_command2
eval $my_command2

my_command3="python Cindy_demo_gratings.py --net res101 --dataset vg\
                   --checksession 1 --checkepoch 20 --checkpoint 16193 \
                   --cuda --stimuli gratings"
echo $my_command3
eval $my_command3

my_command4="python Cindy_demo_gratings.py --net vgg16 --dataset vg\
                   --checksession 1 --checkepoch 19 --checkpoint 48611 \
                   --cuda --stimuli gratings"
echo $my_command4
eval $my_command4
