import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from schedulefree import AdamWScheduleFree

class VectorField(nn.Module):
    def __init__(self, image_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Linear(image_dim + 1, hidden_dim // 2),  # 128 -> 64
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.enc2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 64 -> 32
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder (expanding path)
        self.dec1 = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),  # 32 -> 64
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.dec2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),  # 64 -> 128
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.final = nn.Linear(hidden_dim, image_dim)  # 128 -> output
        
    def forward(self, t, x):
        # Flatten image if it's not already flat
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Concatenate time and flattened image
        tx = torch.cat([t.reshape(-1, 1), x_flat], dim=1)
        
        # Encoder path
        enc1_out = self.enc1(tx)  # 128
        enc2_out = self.enc2(enc1_out)  # 64
        
        # Decoder path (no skip connections)
        dec1_out = self.dec1(enc2_out)  # 64
        dec2_out = self.dec2(dec1_out)  # 128
        
        # Final layer
        out_flat = self.final(dec2_out)
        out = out_flat.view(*x.shape)
        
        return out

def preload_dataset(image_size=128, device="cuda"):
    """Preload and cache the entire dataset in GPU memory"""
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("ceyda/smithsonian_butterflies", split="train")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to [-1, 1]
    ])
    
    # Process all images at once
    all_images = []
    for example in dataset:
        img_tensor = transform(example['image'].convert('RGB'))
        all_images.append(img_tensor)
    
    # Stack all images into a single tensor and move to GPU
    images_tensor = torch.stack(all_images).to(device)
    print(f"Dataset loaded: {images_tensor.shape} ({images_tensor.element_size() * images_tensor.nelement() / 1024/1024:.2f} MB)")
    
    return TensorDataset(images_tensor)

def train_flow_matching(num_epochs=1000, batch_size=2048, device="cuda", sigma_min=0.001):
    # Preload dataset to GPU
    dataset = preload_dataset(device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    
    # Get input dimensions from first batch
    first_batch = next(iter(dataloader))
    image_shape = first_batch[0].shape[1:]  # (C, H, W)
    image_dim = torch.prod(torch.tensor(image_shape)).item()
    
    model = VectorField(image_dim=image_dim).to(device)
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=1e-2,
        warmup_steps=100
    )
    optimizer.train()
    
    # Preallocate tensors for efficiency
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x1 = batch[0]
            batch_size = x1.shape[0]
            
            # Sample random times t ∈ [0,1]
            t = torch.rand(batch_size, device=device)
            
            # Sample noise
            x0 = torch.randn_like(x1)
            
            # Compute OT interpolation
            sigma_t = 1 - (1 - sigma_min) * t.reshape(-1, 1, 1, 1)
            mu_t = t.reshape(-1, 1, 1, 1) * x1
            x_t = sigma_t * x0 + mu_t
            
            # Get vector field prediction
            v_t = model(t, x_t)
            
            # Flow matching loss
            target = x1 - (1 - sigma_min) * x0
            loss = torch.mean((v_t - target) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return model

def sample(model, n_samples=128, n_steps=50, image_size=128, device="cuda"):
    model.eval()
    
    # Start from noise
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1 / n_steps
    
    with torch.no_grad():
        # Euler integration
        for i in range(n_steps):
            t = torch.ones(n_samples, device=device) * (i * dt)
            v = model(t, x)
            x = x + v * dt
    
    # Scale back to [0, 1] for visualization
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    return x

if __name__ == "__main__":
    import torchvision.utils as vutils
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check available VRAM before training
    if device == "cuda":
        print(f"Available VRAM before training: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = train_flow_matching(device=device)

    # Save model
    model_state = model.state_dict()
    torch.save(model_state, "model.safetensors")
    
    os.makedirs("samples", exist_ok=True)
    
    for i in range(5):
        samples = sample(model, n_samples=16, device=device)
        vutils.save_image(samples, f"samples/samples_{i}.png", nrow=4, padding=2)
    
    print("Samples saved in 'samples' directory")
    