import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

class VectorField(nn.Module):
    def __init__(self, input_dim, hidden_dim=768):
        super().__init__()
        # MLP layers for processing input features
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time step
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.sigma_min = 1e-4  # As used in paper

    def forward(self, x, t):
        """
        Forward pass - just predicts vector field
        Args:
            x: Input features (B, D) where D is flattened dimension
            t: Time steps (B,)
        """
        # Reshape if needed (for images)
        batch_size = x.shape[0]
        original_shape = x.shape
        x_flat = x.view(batch_size, -1)
        
        # Combine input and time step
        x_t = torch.cat([x_flat, t.view(-1, 1)], dim=-1)
        
        # Predict vector field
        v_t = self.net(x_t)
        
        # Reshape back to original dimensions
        v_t = v_t.view(original_shape)
        return v_t

def get_dataloader(batch_size=128, image_size=256):
    dataset = load_dataset("reach-vb/pokemon-blip-captions", split="train")
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

def train_matcha_flow(num_epochs=25, batch_size=128, device="cuda"):
    # Get dataloader first to determine input dimensions
    dataloader = get_dataloader(batch_size)
    first_batch = next(iter(dataloader))
    image_shape = first_batch['pixel_values'].shape[1:]  # (C, H, W)
    image_dim = torch.prod(torch.tensor(image_shape)).item()

    # Initialize model and move to device
    model = VectorField(input_dim=image_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x1 = batch['pixel_values'].to(device)
            batch_size = x1.shape[0]

            # Sample random times t ∈ [0,1]
            t = torch.rand(batch_size, device=device)
            
            # Sample initial noise
            x0 = torch.randn_like(x1)
            
            # Compute OT interpolation (Matcha-style)
            x_t = (1 - (1 - model.sigma_min) * t.reshape(-1, 1, 1, 1)) * x0 + t.reshape(-1, 1, 1, 1) * x1
            
            # Get vector field prediction
            v_t = model(x_t, t)
            
            # Compute target (x1 - (1-σ)x0)
            target = x1 - (1 - model.sigma_min) * x0
            
            # Calculate loss
            loss = F.mse_loss(v_t, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # Save a sample every epoch
        if (epoch + 1) % 1 == 0:
            samples = sample_matcha(model, device=device)
            os.makedirs("samples", exist_ok=True)
            vutils.save_image(samples, f"samples/epoch_{epoch}.png", nrow=4, padding=2)
    
    return model

def sample_matcha(model, n_samples=16, n_steps=10, image_size=256, device="cuda"):
    model.eval()
    # Start from noise
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1 / n_steps
    
    with torch.no_grad():
        # Euler integration
        for i in range(n_steps):
            t = torch.ones(n_samples, device=device) * (1 - i * dt)  # Note: reverse time direction
            v = model(x, t)
            x = x + v * dt
    
    # Scale back to [0, 1] for visualization
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Train model
    model = train_matcha_flow(device=device)
    
    # Generate final samples
    for i in range(5):
        samples = sample_matcha(model, n_samples=16, device=device)
        vutils.save_image(samples, f"samples/final_samples_{i}.png", nrow=4, padding=2)
    
    print("Training complete! Samples saved in 'samples' directory")