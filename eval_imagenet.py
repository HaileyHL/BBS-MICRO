import os
import argparse
import time
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from quantize import quantize_model

# --- Configuration ---
IMAGENET_VAL_DIR = os.path.expanduser(
    "/Users/haileyli/Hailey/UCSD/Courses/CSE 240/PTQ4ViT/datasets/imagenet/val")  # adjust if needed
SUBSET_SIZE = 300  # change to 50000 for full validation

# --- Transform Pipelines ---
base_transform = transforms.Resize(256)
single_crop_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
ten_crop = transforms.TenCrop(224)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
vit_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Dataset & Loader ---
full_val_dataset = datasets.ImageFolder(root=IMAGENET_VAL_DIR, transform=base_transform)
val_dataset = Subset(full_val_dataset, list(range(SUBSET_SIZE))) if SUBSET_SIZE < len(full_val_dataset) else full_val_dataset

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), torch.tensor(targets, dtype=torch.long)

val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=collate_fn
)

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model, loader, device, use_tencrop=False, use_vit_transform=False):
    model.eval()
    correct, total = 0, 0

    for imgs_resized, targets in loader:
        if use_vit_transform:
            batch_tensors = torch.stack([vit_transform(img) for img in imgs_resized]).to(device)
            outputs = model(batch_tensors)
            preds = outputs.argmax(dim=1)
        elif use_tencrop:
            all_avg_logits = []
            for img in imgs_resized:
                crops = ten_crop(img)
                crop_tensors = torch.stack([normalize(to_tensor(c)) for c in crops]).to(device)
                avg_logits = model(crop_tensors).mean(dim=0, keepdim=True)
                all_avg_logits.append(avg_logits)
            batch_logits = torch.cat(all_avg_logits, dim=0)
            preds = batch_logits.argmax(dim=1)
        else:
            batch_tensors = torch.stack([single_crop_transform(img) for img in imgs_resized]).to(device)
            outputs = model(batch_tensors)
            preds = outputs.argmax(dim=1)

        correct += (preds == targets.to(device)).sum().item()
        total += targets.size(0)
        print(f"[batch {total}/{len(loader.dataset)}]", end="\r")

    print()
    return 100.0 * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Using {len(val_dataset)} validation images")
    print(f"Running inference on: {device}")

    model_list = [
        ("vgg16", models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, True, False),
        ("resnet34", models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, False, False),
        ("resnet50", models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, False, False),
    ]

    for name, constructor, weight_enum, use_tencrop, use_vit_transform in model_list:
        print(f"\n=== Evaluating {name.upper()} (TenCrop={use_tencrop}, ViT={use_vit_transform}) ===")

        model_fp32 = constructor(weights=weight_enum).to(device)

        acc_fp32 = evaluate(model_fp32, val_loader, device, use_tencrop, use_vit_transform)
        print(f"{name} FP32 Top-1 Acc: {acc_fp32:.2f}%")

        print(f"\n--- Quantizing {name} ---")
        model_int8 = quantize_model(model_fp32, val_loader, single_crop_transform, device="cpu")

        acc_int8 = evaluate(model_int8, val_loader, torch.device("cpu"), use_tencrop, use_vit_transform)
        print(f"{name} INT8 Top-1 Acc (CPU): {acc_int8:.2f}%")
