import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

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

def get_dataloader(batch_size=128, image_size=128):
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to [-1, 1]
    ])
    
    def preprocess(examples):
        images = [transform(image.convert('RGB')) for image in examples['image']]
        return {'pixel_values': torch.stack(images)}
    
    dataset = dataset.with_transform(preprocess)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def train_flow_matching(num_epochs=10, batch_size=128, device="cuda", sigma_min=0.001):
    # Get dataloader first to determine input dimensions
    dataloader = get_dataloader(batch_size)
    first_batch = next(iter(dataloader))
    image_shape = first_batch['pixel_values'].shape[1:] # (C, H, W)
    image_dim = torch.prod(torch.tensor(image_shape)).item()

    # Initialize model and move to device
    model = VectorField(image_dim=image_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x1 = batch['pixel_values'].to(device)
            batch_size = x1.shape[0]

            # Sample random times t ∈ [0,1] 
            t = torch.rand(batch_size, device=device)
            
            # Sample noise
            x0 = torch.randn_like(x1)
            
            SIGMA_MAX = 20000
            # Compute OT interpolation (from section 4.1 of paper)
            sigma_t = sigma_min + t.reshape(-1, 1, 1, 1) * (SIGMA_MAX - sigma_min)
            mu_t = t.reshape(-1, 1, 1, 1) * x1
            x_t = sigma_t * x0 + mu_t
            
            # Get vector field prediction
            v_t = model(t, x_t)
            
            # Flow matching loss (equation 23 in paper)
            target = x1 / SIGMA_MAX - x0
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

    model = train_flow_matching(device=device)
    
    os.makedirs("samples", exist_ok=True)
    
    for i in range(5):
        samples = sample(model, n_samples=16, device=device)
        vutils.save_image(samples, f"samples/samples_{i}.png", nrow=4, padding=2)
    
    print("Samples saved in 'samples' directory")