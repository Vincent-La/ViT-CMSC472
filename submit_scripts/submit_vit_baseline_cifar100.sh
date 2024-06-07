#!/bin/bash
#!/bin/bash

#SBATCH --job-name=vit_baseline_cifar100                        # sets the job name
#SBATCH --output=vit_baseline_cifar100.out.%j                   # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=vit_baseline_cifar100.out.%j                    # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=20:00:00                                         # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger                                  # set QOS, this will determine what resources can be requested
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000

#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              
#SBATCH --ntasks-per-node=1                                     # request 1 cpu core be reserved per node
#SBATCH --mem=16gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed


source ~/.bashrc
# eval "$(micromamba shell hook --shell bash)"
micromamba activate VIT

python train_full_precision.py --name baseline \
                               --model_type=ViT-B_16 \
                               --dataset cifar100 \
                               --pretrained_dir=/fs/nexus-scratch/vla/ViT_pretrained_checkpoints/ViT-B_16.npz \
                               --num_steps 25000 \
                               --train_batch_size=256 \
                               --eval_every 2000

wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
