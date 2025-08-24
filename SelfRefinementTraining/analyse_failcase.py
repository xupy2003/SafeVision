import json
import pandas as pd
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from openai import OpenAI
import requests
import base64
# from lmdeploy import pipeline
# from lmdeploy.vl import load_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy",
    type=str,
    default="./policy.jsonl",
    help="path to test results.",
)
parser.add_argument(
    "--valgroundtruth",
    type=str,
    default="./ground_truth.csv",
    help="path to test ground truth file.",
)

parser.add_argument(
    "--detail",
    type=str,
    default="./details.csv",
    help="path to test ground truth file.",
)

parser.add_argument(
    "--output",
    type=str,
    default="./policy.jsonl",
    help="path to test ground truth file.",
)


args = parser.parse_args()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

policies = {}
with open(args.policy, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        policies[data["class"]]=data["content"]

details = pd.read_csv(args.detail)
test_set = pd.read_csv(args.valgroundtruth)

for index, item in details.iterrows():
    id = item["ID"]
    print(f"analyzing image {id}")
    gt_label = item["Ground Truth"]
    test_row = test_set[test_set["ID"] == id]
    if not test_row.empty:
        test_row = test_row.iloc[0]
        image_path = test_row["Image Path"]
        target_policy = policies[gt_label]
        category = ""
        if gt_label == "[0]":
            category = "Safe"
        elif gt_label == "[1]":
            category = "Hate_Humiliation_Harassment"
        elif gt_label == "[2]":
            category = "Violence_Harm_Cruelty"
        elif gt_label == "[3]":
            category = "Sexual"
        elif gt_label == "[4]":
            category = "Criminal_Planning"
        elif gt_label == "[5]":
            category = "Weapons_Substance_Abuse"
        elif gt_label == "[6]":
            category = "Self_Harm"
        elif gt_label == "[7]":
            category = "Animal_Cruelty"
        elif gt_label == "[8]":
            category = "Disasters_Emergencies"
        elif gt_label == "[9]":
            category = "Political"
        question = f'''
You are an image moderation assistant. Your task is to help modify the moderation policy based on the given image.
The given image is categorized to {category}.

The moderation policy is:
####
{target_policy}
####

- If you think the image fits the given moderation policy, reply "No change";
- If you believe the image should not be classified into this category, ONLY reply "No change." DO NOT change the policy or provide an explanation as to why the image is misclassified;
- If you think the image should be classified into this category, but the given moderation policy is not comprehensive enough to cover the situation of this image, please modify the moderation policy and return the modified moderation policy.
You MUST make change to the original moderation policy! The format of the moderation policy you return MUST be :
{category}
Core Value: xxx
This class includes but is not limited to:
- xxx
- xxx
- xxx

DO NOT include any other information!
Now please strictly follow the instruction above and give your response.
'''
        base64_image = encode_image(image_path)
        headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer xxx"
                }

        payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": question
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2048,
                    "temperature": 0
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # print(response.json())
        responses = [resp['message']['content'] for resp in response.json()['choices']]
        
        response = responses[0]
        if "no change" in response.lower():
            continue
        else:
            policies[gt_label] = response
            with open(args.output, 'w', encoding='utf-8') as outfile:
                for key, value in policies.items():
                    json.dump({"class": key, "content": value}, outfile)
                    outfile.write('\n')

    else:
        continue