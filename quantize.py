import torch

def quantize_model(model_fp32, val_loader, preprocess_transform, device='cpu'):
    """
    Post-training static per-channel quantization (8-bit).
    Returns a quantized model ready for evaluation.
    """
    model = model_fp32.to(device).eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    model = torch.ao.quantization.prepare(model, inplace=False)
    # Calibration: run a few batches through the model
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = [preprocess_transform(img) for img in imgs]
            batch = torch.stack(imgs).to(device)
            model(batch)
            break  # One batch is usually enough for calibration
    model = torch.ao.quantization.convert(model, inplace=False)
    return model