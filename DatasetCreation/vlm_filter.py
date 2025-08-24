import torch
from PIL import Image
from transformers import AutoProcessor
import torch.nn as nn
import os
from torchvision import transforms
from transformers import AutoProcessor, AutoModel
import pandas as pd
import requests
from io import BytesIO
from tqdm import tqdm 
import argparse
import csv   
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import json

def process_files(file_paths, output_path):
    id_split_result_counts = {}

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                id = entry["ID"]
                split = entry["split"]
                result = entry["result"].lower()
                key = (id, split)
                if key not in id_split_result_counts:
                    id_split_result_counts[key] = 0
                if "no" in result:
                    id_split_result_counts[key] += 1

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for (id, split), count in id_split_result_counts.items():
            if count >= 4:
                result_entry = {"ID": id, "times": count, "split": split}
                output_file.write(json.dumps(result_entry) + '\n')
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="./image",
        help="path to test ground truth file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./groundtruth.csv",
        help="path to test ground truth file.",
    )    
    parser.add_argument(
        "--policy",
        type=str,
        default="./policy.jsonl",
        help="path to test ground truth file.",
    ) 
    args = parser.parse_args()
    image_id = 0
    with open(args.output, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ID', 'Label', 'Source', 'Image Path'])
        for i in range(1, 10):
            label = f'[{i}]'
            folder_path = os.path.join(args.image, str(i))
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): 
                    image_path = os.path.join(folder_path, filename)
                    absolute_image_path = os.path.abspath(image_path)
                    csvwriter.writerow([image_id, label, 'LAION-400M', absolute_image_path])
                    image_id += 1

    print(f'Successfully generated groundtruth.csv , include {image_id} items')

    script_paths = ['intern8b.py','intern26b.py','qwen_filter.py','LLaVA/llava_filter.py']
    output_files = ['results/intern8b.jsonl', 'results/intern26b.jsonl', 'results/qwen.jsonl', 'results/llava.jsonl']

    def run_script(script_path, output_file):
        output_path = output_file
        arguments = [
            '--groundtruth', args.output,
            '--policy', args.policy,
            '--output', output_path
        ]

        command = ['python', script_path] + arguments
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        return script_path, process.returncode, stdout, stderr

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_script, script_path, output_file) for script_path, output_file in zip(script_paths, output_files)]
        
        for future in as_completed(futures):
            script_path, returncode, stdout, stderr = future.result()
    
    if returncode != 0:
        print(f"Script {script_path} failed with return code {returncode}")
        print("Output:", stdout)
        print("Error:", stderr)

        for f in futures:
            if not f.done():
                f.cancel()
        
        sys.exit(1)

    print(f"Script: {script_path}")
    print("Return code:", returncode)
    print("Output:", stdout)
    print("Error (if any):", stderr)
    
    file_paths = [output_files[0],output_files[1], output_files[2], output_files[3]]
    output_path = "results/results.json"
    process_files(file_paths, output_path)
    print(f"Results written to {output_path}")
    
    ## filter dataset
    df = pd.read_csv(args.output)
    results = []
    with open(output_path, 'r', encoding='utf-8') as file:
        for line in file:
            results.append(json.loads(line))
    for result in results:
        id_to_remove = result["ID"]
        df = df[df["ID"] != id_to_remove]

    df.to_csv(args.output, index=False)
    print("Filtered datasets have been saved.")


if __name__ == "__main__":
    main()
