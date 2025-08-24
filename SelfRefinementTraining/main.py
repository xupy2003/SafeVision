import torch
import os
from PIL import Image
import json
import re
import argparse
import pandas as pd
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import sys
import math

parser = argparse.ArgumentParser()
parser.add_argument(
    "--valgroundtruth",
    type=str,
    default="./ground_truth.csv",
    help="path to test ground truth file.",
)
parser.add_argument(
    "--traingroundtruth",
    type=str,
    default="./train_ground_truth.csv",
    help="path to train ground truth file.",
)
parser.add_argument(
    "--policy",
    type=str,
    default="./policy.jsonl",
    help="path to load basic policy.",
)
parser.add_argument(
    "--model",
    type=str,
    default="SafeVision-8B",
    help="path to load model.",
)
parser.add_argument(
    "--finetune_dataset",
    type=str,
    default="./test.jsonl",
    help="path to load finetune dataset.",
)

args = parser.parse_args()


def process_files(file_paths, output_path, round):
    weight_our_model = 0.15 * math.sqrt(round)
    weight_others = (1 - weight_our_model) / 3
    weights = [weight_our_model, weight_others, weight_others, weight_others]
    id_split_scores = {}
    for i, file_path in enumerate(file_paths):
        weight = weights[i]  
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                id = entry["ID"]
                split = entry["split"]
                result = entry["result"].lower()
                key = (id, split)
                result_value = 1 if "no" in result else 0
                if key not in id_split_scores:
                    id_split_scores[key] = 0
                id_split_scores[key] += result_value * weight
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for (id, split), score in id_split_scores.items():
            if score > 0.7 + 0.05 * round:
                result_entry = {"ID": id, "split": split, "score": score}
                output_file.write(json.dumps(result_entry) + '\n')




def reformat_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            image_path = data["image"]
            conversations = data["conversations"]

            for i in range(0, len(conversations), 2):
                if conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt":
                    query = conversations[i]["value"].replace("<image>\n\n", "")
                    response = conversations[i+1]["value"]
                    new_entry = {
                        "query": query,
                        "response": response,
                        "images": [image_path]
                    }
                    outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                    
                    
def main():
    round = 1
    our_model_pth = args.model
    test_ground_truth = args.valgroundtruth
    train_ground_truth = args.traingroundtruth
    policy_path = args.policy
    finetune_dataset_path = args.finetune_dataset
    previous_accuracy = 0.0
    while True:
        # load the basic data-filtering policy
        policy = []
        with open(policy_path, 'r', encoding='utf-8') as f:
            for line in f:
                policy.append(json.loads(line))
        parent_folder = 'results'
        os.makedirs(parent_folder, exist_ok=True)
        sub_folder_name = f'round_{round}_results'
        sub_folder_path = os.path.join(parent_folder, sub_folder_name)
        os.makedirs(sub_folder_path, exist_ok=True)
        
        
        # use 4 models to filter the dataset
        
        # generate filter results
        script_paths = ['our_model.py','intern26b.py','qwen_filter.py','LLaVA/llava_filter.py']
        output_files = ['our_model.jsonl', 'intern26b.jsonl', 'qwen.jsonl', 'llava.jsonl']

        def run_script(script_path, output_file):
            output_path = os.path.join(sub_folder_path, output_file)
            arguments = []
            if script_path == 'our_model.py':
                arguments = [
                    '--model', our_model_pth,
                    '--traingroundtruth', train_ground_truth,
                    '--policy', policy_path,
                    '--output', output_path
                ]
            else:
                arguments = [
                    '--model', our_model_pth,
                    '--traingroundtruth', train_ground_truth,
                    '--policy', policy_path,
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
        
        ## process filter results
        file_paths = [os.path.join(sub_folder_path, output_files[0]), os.path.join(sub_folder_path, output_files[1]), os.path.join(sub_folder_path, output_files[2]), os.path.join(sub_folder_path, output_files[3])]
        output_path = os.path.join(sub_folder_path, "results.json")
        process_files(file_paths, output_path, round)
        print(f"Results written to {output_path}")
        
        ## filter dataset
        train_df = pd.read_csv(train_ground_truth)
        test_df = pd.read_csv(test_ground_truth)
        results = []
        with open(output_path, 'r', encoding='utf-8') as file:
            for line in file:
                results.append(json.loads(line))
        for result in results:
            id_to_remove = result["ID"]
            split = result["split"]
            if split == "train":
                train_df = train_df[train_df["ID"] != id_to_remove]

        filtered_train_ground_truth = os.path.join(sub_folder_path, "train_ground_truth.csv")
        filtered_test_ground_truth = os.path.join(sub_folder_path, "test_ground_truth.csv")
        train_df.to_csv(filtered_train_ground_truth, index=False)
        test_df.to_csv(filtered_test_ground_truth, index=False)
        test_ground_truth = filtered_test_ground_truth
        train_ground_truth = filtered_train_ground_truth
        print("Filtered datasets have been saved.")

        ## filter finetune dataset
        train_df = pd.read_csv(filtered_train_ground_truth)
        train_ids = set(train_df["ID"].tolist())
        filtered_results = []
        with open(finetune_dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                if int(entry["id"]) in train_ids:
                    filtered_results.append(entry)
        filtered_finetune_dataset_path = os.path.join(sub_folder_path, "test_filter.jsonl")
        with open(filtered_finetune_dataset_path, 'w', encoding='utf-8') as output_file:
            for entry in filtered_results:
                output_file.write(json.dumps(entry) + '\n')
        print("Filtered test.jsonl has been saved to test_filter.jsonl.")
        finetune_dataset_path = filtered_finetune_dataset_path
        
        ## generate swift finetune dataset
        swift_finetune_path = os.path.join(sub_folder_path, "swift_filter.jsonl")
        reformat_jsonl(filtered_finetune_dataset_path, swift_finetune_path)
        
        
        
        # start lora finetuning
        if os.path.exists('output'):
            shutil.rmtree('output')
            print(f"Deleted output directory.")
        swift_finetune_path = os.path.join(sub_folder_path, "swift_filter.jsonl")
        script = f"""
        NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=1,3,4,5 swift sft \
        --model_type internvl2-8b \
        --dataset {swift_finetune_path} \
        --max_length 4096 \
        --model_id_or_path /scratch/xpy/image_moderation/InternVL/internvl_chat/pretrained/InternVL2-8B \

        """
        # --deepspeed default-zero2
        result = subprocess.run(script, shell=True)
        
        root_dir = 'output/internvl2-8b'
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"The directory {root_dir} does not exist.")
        subfolders = next(os.walk(root_dir))[1]
        if len(subfolders) != 1:
            raise ValueError(f"Expected exactly one subfolder in {root_dir}, found {len(subfolders)}.")
        subfolder = subfolders[0]
        checkpoint_dir = os.path.join(root_dir, subfolder)
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        if not checkpoints:
            raise ValueError(f"No checkpoint directories found in {checkpoint_dir}.")
        max_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        final_path = os.path.join(checkpoint_dir, max_checkpoint)
        script = f"""CUDA_VISIBLE_DEVICES=1 swift export --ckpt_dir {final_path} --merge_lora true"""
        result = subprocess.run(script, shell=True)
        merged_path = final_path+'-merged'
        destination_path = os.path.join(sub_folder_path, os.path.basename(merged_path))
        shutil.copytree(merged_path, destination_path)
        print(f"Copied {merged_path} to {destination_path}")
        
        
        
        # test finetuned model on the test dataset  and get the result
        model_path = merged_path
        our_model_pth = merged_path
        # model_path = '/scratch/xpy/image_moderation/automatic_pipeline/output/internvl2-8b/v0-20240805-234650/checkpoint-8464-merged'
        script_path = 'test_dataset.py'
        script = f'''
        python {script_path} --groundtruth {sub_folder_path+'/test_ground_truth.csv'} --model {model_path} --output {sub_folder_path+'/test_result.json'} --policy {policy_path}
        '''
        subprocess.run(script, shell=True)
        
        
        
        # analyse the result:
        script_path = 'analyse_result.py'
        script = f'''
        python {script_path} --resultfile {sub_folder_path+'/test_result.json'} --valgroundtruth {sub_folder_path+'/test_ground_truth.csv'}  --output {sub_folder_path+'/result_analyse.json'}  --detail {sub_folder_path+'/details.csv'}
        '''
        subprocess.run(script, shell=True)  
        
        with open(sub_folder_path+'/result_analyse.json', 'r') as f:
            data = json.load(f)
        accuracy = data["multi-class acc"]
        if accuracy < previous_accuracy or (accuracy-previous_accuracy) <= 0.0001 :
            break
        else:
            previous_accuracy = accuracy
        
        #  analyze the fail cases and update the basic policy
        script_path = 'analyse_failcase.py'
        script = f'''
        python {script_path} --policy {policy_path} --valgroundtruth {sub_folder_path+'/test_ground_truth.csv'}  --output {sub_folder_path+'/policy.json'}  --detail {sub_folder_path+'/details.csv'}
        '''
        subprocess.run(script, shell=True) 
        policy_path = sub_folder_path+'/policy.json'
        round += 1
    
if __name__ == "__main__":
    main()