# Dataset Generation Process
## Step1: fine-tune image classification model
```
python finetune_vit.py --train xx --test xx --dataset xx
```
## Step2: use fine-tuned image classification model to filter laion dataset
```
python vit_filter.py --model xxx --laiondata xxx
```

## Step3: use VLM to fuether filter laion dataset
```
python vlm_filter.py --image xxx --output xxx --policy xxx
```

## Step4: generate fine-tune dataset
```
python gen_ft_dataset.py --groundtruth xxx --output xxx --model xxx
```