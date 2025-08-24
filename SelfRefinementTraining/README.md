# Self-Refinement Training Stage

script to start self-refinement training: `script -c "python main.py --valgroundtruth xxx --traingroundtruth xxx --policy xxx --model xxx --finetune_dataset xxx" results/log.txt`

- --valgroundtruth: path to validation ground truth
- --traingroundtruth: path to training ground truth
- --policy: path to the policy file obtained in the model preparation stage
- --model: path to our model
- --finetune_dataset: path to the finetune dataset obtained in the dataset creation stage