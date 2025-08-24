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

class CustomModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(1152, num_labels)

    def forward(self, pixel_values):
        outputs = self.base_model.vision_model(pixel_values=pixel_values)
        logits = self.classifier(outputs.pooler_output)
        return logits
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="fine-tuned-siglip-model_12.pth",
        help="path to test ground truth file.",
    )
    parser.add_argument(
        "--laiondata",
        type=str,
        default="/scratch/xpy/image_moderation/laion-400m/laion400m-meta/nsfw_only.parquet",
        help="path to test ground truth file.",
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

    num_labels = 10
    model = CustomModel(base_model, num_labels)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", do_rescale=False)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    df = pd.read_parquet(args.laiondata)
    total_rows = len(df)

    for index, row in tqdm(df.iloc[0:total_rows].iterrows(), total = total_rows, desc="Processing Images"):
        url = row['URL']       
        try:
            response = requests.get(url, timeout=2) 
            original_image = Image.open(BytesIO(response.content)).convert("RGB")  
            image = transform(original_image) 
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs["pixel_values"]

            with torch.no_grad():
                logits = model(pixel_values=pixel_values)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                predicted_prob = probabilities[0, predicted_class].cpu().numpy()
                
            if predicted_class not in [0] and predicted_prob > 0.6:
                save_dir = f"./image/{predicted_class}/"
                os.makedirs(save_dir, exist_ok=True)
                original_image.save(os.path.join(save_dir, f"{index}.jpg")) 
        
        except requests.exceptions.Timeout:
            print(f"Timeout for URL: {url}")
            continue 
        
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            continue 

if __name__ == "__main__":
    main()
