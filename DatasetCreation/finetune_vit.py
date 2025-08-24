import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, AutoModel, AdamW
from sklearn.metrics import accuracy_score
import torch.nn as nn
import os
from tqdm import tqdm  

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    type=str,
    default="./train_ground_truth.csv",
    help="path to train ground truth file.",
)
parser.add_argument(
    "--test",
    type=str,
    default="./test_ground_truth.csv",
    help="path to test ground truth file.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="./test_ground_truth.csv",
    help="path to the dataset.",
)
args = parser.parse_args()

train_df = pd.read_csv(args.train)
test_df = pd.read_csv(args.test)
label_mapping = {f"[{i}]": i for i in range(10)}
train_df["Label"] = train_df["Label"].map(label_mapping)
test_df["Label"] = test_df["Label"].map(label_mapping)

class CustomDataset(Dataset):
    def __init__(self, dataframe, processor, transform=None):
        self.dataframe = dataframe
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["Image Path"]
        dataset_path = args.dataset
        img_path = os.path.join(dataset_path, img_path)
        label = self.dataframe.iloc[idx]["Label"] 
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        
        return {"image": image, "label": label}

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", do_rescale=False) 

train_dataset = CustomDataset(train_df, processor, transform)
test_dataset = CustomDataset(test_df, processor, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

base_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
for param in base_model.parameters():
    param.requires_grad = False

class CustomModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(1152, num_labels) 

    def forward(self, pixel_values, labels=None):
        outputs = self.base_model.vision_model(pixel_values=pixel_values)
        logits = self.classifier(outputs.pooler_output) 

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return logits, loss

num_labels = 10
model = CustomModel(base_model, num_labels)
model.to(device)  
model.train()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for batch in epoch_iterator:
        images = batch["image"]
        labels = batch["label"].to(device) 

        inputs = processor(images=images, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]

        logits, loss = model(pixel_values=pixel_values, labels=labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_iterator.set_postfix(loss=epoch_loss/len(epoch_iterator))

    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"]
            labels = batch["label"].to(device)  

            inputs = processor(images=images, return_tensors="pt").to(device)
            pixel_values = inputs["pixel_values"]
            logits, _ = model(pixel_values=pixel_values)

            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            truths.extend(labels.cpu().numpy())

    acc = accuracy_score(truths, preds)
    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {acc:.4f}")
    if epoch >= 10:
        torch.save(model.state_dict(), f"fine-tuned-siglip-model_{epoch}.pth")
