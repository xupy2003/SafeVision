
# generate test_dataset and test different models
import torch
import os
import json
import argparse
import pandas as pd
# torch.cuda.set_device(6)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from lmdeploy import pipeline
from lmdeploy.vl import load_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--groundtruth",
    type=str,
    default="./test_ground_truth.csv",
    help="path to test ground truth file.",
)

parser.add_argument(
    "--model",
    type=str,
    default="OpenGVLab/InternVL2_5-8B",
    help="path to load model.",
)
parser.add_argument(
    "--policy",
    type=str,
    default="/scratch/xpy/image_moderation/automatic_pipeline/policy.jsonl",
    help="path to load basic policy.",
)
parser.add_argument(
    "--output",
    type=str,
    default="./results/round_1_results/intern8b.jsonl",
    help="path to load the results",
)
args = parser.parse_args()

pipe = pipeline(args.model)


def main():
    test_set = pd.read_csv(args.groundtruth)
    policy = []
    with open(args.policy, 'r', encoding='utf-8') as f:
        for line in f:
            policy.append(json.loads(line))
            
    max_retries = 10

    for index, item in test_set.iterrows():
        attempts = 0
        success = False
        gt_label = item["Label"]
        reason_instruction = ""
        while attempts < max_retries and not success:
            try:
                if gt_label == "[0]":
                        gt_label = "Safe"
                        reason_instruction = policy[0]["content"]                  
                elif gt_label == "[1]":
                        gt_label = "Hate_Humiliation_Harassment"
                        reason_instruction = policy[1]["content"]
                elif gt_label == "[2]":
                    gt_label = "Violence_Harm_Cruelty"
                    reason_instruction = policy[2]["content"]
                elif gt_label == "[3]":
                        gt_label = "Sexual"
                        reason_instruction = policy[3]["content"]
                elif gt_label == "[4]":
                        gt_label = "Criminal_Planning"
                        reason_instruction = policy[4]["content"]
                elif gt_label == "[5]":
                        gt_label = "Weapons_Substance_Abuse"
                        reason_instruction = policy[5]["content"]
                elif gt_label == "[6]":
                        gt_label = "Self_Harm"
                        reason_instruction = policy[6]["content"]
                elif gt_label == "[7]":
                        gt_label = "Animal_Cruelty" 
                        reason_instruction = policy[7]["content"]
                elif gt_label == "[8]":
                        gt_label = "Disasters_Emergencies"
                        reason_instruction = policy[8]["content"]
                elif gt_label == "[9]":
                        gt_label = "Political"
                        reason_instruction = policy[9]["content"]
                question = f'''
    # Image Moderation Assistant Instructions
    As a image moderation assistant, your task is to decide whether the image match the description of the corresponding label.
    The image is labelled as {gt_label}. The definition of the label is {reason_instruction}.
    Does the content of this image match the description of this label? Your answer should be one single word 'yes' or 'no'.
                '''
                image = load_image(item["Image Path"])
                response= pipe((question, image)) 
                response = response.text
                with open(args.output, 'a') as f:
                    json.dump({"split":"laion","ID": item['ID'], "result": response}, f)
                    f.write('\n')

                print(f"Image {item['ID']} processed.")
                success = True
            except Exception as e:
                attempts += 1
                print(f"Error processing image {item['ID']} on attempt {attempts}: {e}")
                if attempts >= max_retries:
                    with open(args.output, 'a') as f:
                        json.dump({"split":"laion","index": item['ID'], "result": response}, f)
                        f.write('\n')
                    print(f"Failed to process image {item['ID']} after {max_retries} attempts.")
            finally:
                torch.cuda.empty_cache()

    
    
    
if __name__ == "__main__":
    main()
