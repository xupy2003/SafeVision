
# generate test_dataset and test different models
import torch
import os
import json
import re
import argparse
import pandas as pd
import requests
import base64
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from openai import OpenAI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--rawfile",
    type=str,
    default="./sample.txt",
    help="path to ground truth file.",
)
parser.add_argument(
    "--outputfile",
    type=str,
    default="./policy.jsonl",
    help="path to ground truth file.",
)
args = parser.parse_args()


def extract_and_load_json(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_string = json_match.group(0)
        data = json.loads(json_string)
        return data
    else:
        raise ValueError("No JSON found in the text")





def main():
    content = ""
    with open(args.rawfile, 'r', encoding='utf-8') as file:
        content = file.read()  

    max_retries = 1
    attempts = 0
    success = False
    parse_prompt = f'''
    Your task is to generate a specific policy in json format based on a given phase.
    <OUTPUT FORMAT>
    {{"class": "xxx", "content": "xxx"}} 
    <OUTPUT FORMAT>
    
    <EXAMPLES BEGIN>
    
    # Example 1 #
    <INPUT>
    The Hate, Humiliation, and Harassment category encompasses images containing hate speech, gestures, or content that humiliates or harasses individuals or groups. This includes images with hateful symbols, negative stereotypes, or depictions that dehumanize certain groups by comparing them to animals or objects. Examples include symbols such as the swastika, Confederate flag, SS runes, KKK hood and cross, Celtic cross, anti-Semitic or anti-Muslim symbols, anti-Black symbols, and white power insignia. It also covers imagery that mocks, threatens, or bullies individuals through actions like sexual harassment, threatening messages, cyberbullying, workplace harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, and stalking. Content in this category specifically targets individuals or groups based on race, ethnicity, gender, religion, or other personal characteristics.
    <OUTPUT>
    {{"class": "<|Hate_Humiliation_Harassment|>", "content": "Core Value: All the images that contain hate speech or gestures, humiliation, or harassment. This class includes but is not limited to: - images that include hateful symbols, negative stereotypes, comparing certain groups to animals/objects - images expressing or promoting hate based on identity. For example, swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols, anti-Black symbols, white power symbols, etc. - images that shows mocking, threatening, or bullying an individual. For example, sexual harassment, threatening messages, cyberbullying, harassment in the workplace, online harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, being stalked, etc. - images that target individuals or groups based on race, ethnicity, gender, religion, or other characteristics"}} 
    
    # Example 2 #
    <INPUT> 
    The Violence, Harm, and Cruelty category includes all images that depict acts of violence, harm, or cruelty. This category encompasses, but is not limited to, images showing violent acts such as torture, homicide, war and combat scenes, mass shootings, and child abuse. It also includes images of physical fights or assaults, depictions of injuries or harm inflicted on individuals, and scenes of people planning or actively engaging in violent behavior. 
    <OUTPUT> 
    {{"class": "<|Violence_Harm_Cruelty|>", "content": "Core Value: All the images that contain violence, harm, or cruelty. This class includes but is not limited to: - images that show violent acts like torture, homicide, war and combat scenes, mass shootings, child abuse, etc. - images of physical fights or assaults - images depicting injuries or harm to individuals - images of people planning or engaging in violence"}} 
    
    # Example 3 #
    <INPUT>
    The Sexual category encompasses all images containing explicit sexual content. This includes, but is not limited to, images that depict sexual acts or other explicit content intended to be erotic, images showing sexually suggestive poses or attire, and those designed to arouse sexual interest. Additionally, it covers images displaying visible genitalia (such as exposed breasts, nipples, vaginas, vulvas, or penises) or explicit nudity. Importantly, this category excludes images where nudity is presented in a contextually appropriate educational or artistic manner.
    <OUTPUT>
    {{"class": "<|Sexual|>", "content": "Core Value: All the images that contain explicit sexual content. This class includes but is not limited to: - images depicting sexual acts and oher sexually explicit (i.e., erotic) content - images showing sexually suggestive poses or attire - images intended to arouse sexual interest - images that contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content - images where the nudity is not contextualized in an educational or artistic manner"}} 
    <EXAMPLES END>
    
    Now I will provide you with the target phase, your answer should ONLY include a policy in json format.
    <INPUT>
    {content}
    <OUTPUT>
    '''
    while attempts < max_retries and not success:
        try:
            question = parse_prompt
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
                            }
                        ]
                    }
                ],
                "max_tokens": 8192,
                "temperature": 0
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            responses = [resp['message']['content'] for resp in response.json()['choices']]
            
            response = responses[0]
            print(response)
            response = extract_and_load_json(text=response)
            with open(args.outputfile, 'a') as f:
                json.dump(response, f)
                f.write('\n')
            success = True
        except Exception as e:
            attempts += 1
            print(f"Error processing policy on attempt {attempts}: {e}")
            if attempts >= max_retries:
                print(f"Failed to process policy  after {max_retries} attempts.")
        finally:
            torch.cuda.empty_cache()
    
    
    
if __name__ == "__main__":
    main()
