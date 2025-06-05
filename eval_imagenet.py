import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import timm
from torchvision.transforms import InterpolationMode

# ─── 1. Base transform for the DataLoader: Resize shorter side to 256 (bilinear by default)
base_transform = transforms.Resize(256)

# Path to your reorganized ImageNet-1K validation directory (50 000 images, 1000 subfolders)
imagenet_val_dir = "/Users/haileyli/Hailey/UCSD/Courses/CSE 240/ILSVRC2012_img_val_reorganized"

# ─── 2. Build the full validation dataset (50 000 images), each as a PIL.Image resized to shorter=256
full_val_dataset = datasets.ImageFolder(
    root=imagenet_val_dir,
    transform=base_transform
)

val_dataset = full_val_dataset  # we want all 50 000

# ─── 3. Custom collate_fn so DataLoader returns (list_of_PIL, tensor_of_targets)
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), torch.tensor(targets, dtype=torch.long)

# ─── 4. Build the DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    collate_fn=collate_fn
)

# ─── 5. Transform for single‐crop (ResNet/VGG) after base_transform
single_crop_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── 6. Transforms for 10‐crop (VGG)
ten_crop = transforms.TenCrop(224)   # yields a tuple of 10 PIL crops
to_tensor   = transforms.ToTensor()
normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

# ─── 7. Transforms for any ViT/DeiT (bicubic Resize→CenterCrop→ToTensor→Normalize)
vit_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def evaluate(model, loader, device, use_tencrop=False, use_vit_transform=False):
    """
    - If use_vit_transform=True: Resize→(bicubic)→CenterCrop(224) → This is for DeiT/ViT.
    - Else if use_tencrop=True:       10‐crop pipeline → This is only for VGG.
    - Else:                           single‐center‐crop → This covers ResNet (and any default).
    """
    model.eval()
    correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (imgs_resized, targets) in enumerate(loader, start=1):
        # imgs_resized: list of PIL.Image, each already resized so shorter side = 256

        if use_vit_transform:
            # ─── DeiT/ViT path: bicubic Resize→CenterCrop(224)
            batch_tensors = torch.stack([
                vit_transform(img) for img in imgs_resized
            ], dim=0).to(device, non_blocking=True)
            outputs = model(batch_tensors)
            preds = outputs.argmax(dim=1)

        elif use_tencrop:
            # ─── VGG’s 10‐crop path: generate 10 crops, normalize, average logits
            all_avg_logits = []
            for img in imgs_resized:
                crops = ten_crop(img)  # tuple of 10 PILs 224×224
                crop_tensors = torch.stack([
                    normalize(to_tensor(c)) for c in crops
                ], dim=0).to(device)  # (10, 3, 224, 224)
                logits10 = model(crop_tensors)              # (10, 1000)
                avg_logits = logits10.mean(dim=0, keepdim=True)  # (1, 1000)
                all_avg_logits.append(avg_logits)
            batch_logits = torch.cat(all_avg_logits, dim=0).to(device)  # (B, 1000)
            preds = batch_logits.argmax(dim=1)

        else:
            # ─── Single‐center‐crop path (ResNet, or default for others)
            batch_tensors = torch.stack([
                single_crop_transform(img) for img in imgs_resized
            ], dim=0).to(device, non_blocking=True)  # (B, 3, 224, 224)
            outputs = model(batch_tensors)               # (B, 1000)
            preds = outputs.argmax(dim=1)                # (B,)

        correct += (preds == targets.to(device)).sum().item()
        total += targets.size(0)

        # Print running progress on one line
        print(f"[batch {batch_idx}/{num_batches}]")

    # Newline after finishing all batches
    print()
    return 100.0 * correct / total


if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # ─── 8. Device selection: prefer MPS on Apple Silicon, then CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Using {len(val_dataset)} validation images")
    print("Running inference on:", device)

    # ─── 9. Build model_list
    # Each entry: ( name,
    #               constructor,            # either torchvision or timm
    #               torchvision_weight_or_False,
    #               use_tencrop,
    #               use_vit_transform )
    model_list = [
        # # VGG16   → 10‐crop to match 73.36 % in the paper
        # ("vgg16",    models.vgg16,    models.VGG16_Weights.IMAGENET1K_V1,    True,  False),

        # # ResNet  → single‐crop (paper numbers use single center‐crop)
        # ("resnet34", models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, False, False),
        # ("resnet50", models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, False, False),

        # # DeiT-Small  → single‐crop Resize(256, bicubic) → CenterCrop(224)  → ~81.8 %
        # ("deit_s",   lambda **kw: timm.create_model("deit_small_patch16_224", pretrained=True, **kw),
        #              False, False, True),

        # ViT-Base (ImageNet-21k→ImageNet-1k) → single‐crop Resize(256, bicubic) → CenterCrop(224) → ~84 %
        ("vit_b",    lambda **kw: timm.create_model("vit_base_patch16_224_in21k", pretrained=True, **kw),
                     False, False, True),
    ]

    # ─── 10. Evaluate each model
    for name, constructor, weight_enum, use_tencrop, use_vit_transform in model_list:
        print(f"\n=== Evaluating FP32 {name} on {device}  "
              f"(TenCrop={use_tencrop}, ViTTransform={use_vit_transform}) ===")

        # Instantiate model:
        if weight_enum is False:
            # timm constructor: the lambda already does pretrained=True
            model = constructor().to(device)
        else:
            # torchvision constructor: pass weights=weight_enum
            model = constructor(weights=weight_enum).to(device)

        start_time = time.time()
        top1 = evaluate(model, val_loader, device,
                        use_tencrop=use_tencrop,
                        use_vit_transform=use_vit_transform)
        elapsed = time.time() - start_time
        print(f"{name} FP32 Top-1 on 50000 images: {top1:.2f}%  (time: {elapsed:.1f} s)")

    # ─── 11. Clean up DataLoader workers so the process exits immediately
    del val_loader
