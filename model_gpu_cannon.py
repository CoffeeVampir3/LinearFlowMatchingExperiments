import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from schedulefree import AdamWScheduleFree

class VectorField(nn.Module):
    def __init__(self, image_dim, hidden_dim=256):
        super().__init__()
        # Input: (t, flattened_image) where t is time and flattened_image is the flattened pixel values
        self.net = nn.Sequential(
            nn.Linear(image_dim + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, image_dim) # In dim == out dim
        )
        self.weight_init()

    def weight_init(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        
        nn.init.normal_(self.net[4].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[4].bias)
    
    def forward(self, t, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        tx = torch.cat([t.reshape(-1, 1), x_flat], dim=1)
        
        out_flat = self.net(tx)
        out = out_flat.view(*x.shape)
        return out

def preload_dataset(image_size=256, device="cuda"):
    """Preload and cache the entire dataset in GPU memory"""
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("reach-vb/pokemon-blip-captions", split="train")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to [-1, 1]
    ])

    # Process all images at once
    all_images = []
    for example in dataset:
        # The Pokemon dataset stores images in 'image' field
        img_tensor = transform(example['image'].convert('RGB'))
        all_images.append(img_tensor)

    # Stack all images into a single tensor and move to GPU
    images_tensor = torch.stack(all_images).to(device)
    print(f"Dataset loaded: {images_tensor.shape} ({images_tensor.element_size() * images_tensor.nelement() / 1024/1024:.2f} MB)")
    
    return TensorDataset(images_tensor)

def train_flow_matching(num_epochs=5000, batch_size=16, device="cuda", sigma_min=0.001):
    # Preload dataset to GPU
    dataset = preload_dataset(device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    
    # Get input dimensions from first batch
    first_batch = next(iter(dataloader))
    image_shape = first_batch[0].shape[1:]  # (C, H, W)
    image_dim = torch.prod(torch.tensor(image_shape)).item()
    
    model = VectorField(image_dim=image_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=0.1)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=100,    # Number of iterations for a complete cosine cycle
        eta_min=0.0001,     # Minimum learning rate
    )

    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x1 = batch[0]
            batch_size = x1.shape[0]
            
            # Sample t uniform in [0,1] 
            t = torch.rand(batch_size, 1, 1, 1, device=device)
            
            # Sample noise from standard normal
            x0 = torch.randn_like(x1)
            
            # Compute OT path interpolation (equation 22)
            sigma_t = 1 - (1 - sigma_min) * t
            mu_t = t * x1
            x_t = sigma_t * x0 + mu_t
            
            # Get vector field prediction
            v_t = model(t.squeeze(), x_t)
            
            # Compute target (equation 23)
            target = x1 - (1 - sigma_min) * x0
            
            # Compute loss
            loss = F.mse_loss(v_t, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return model

def sample(model, n_samples=128, n_steps=50, image_size=256, device="cuda", sigma_min=0.2):
    model.eval()
    
    # Start from pure noise (t=0)
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    
    # Time steps from t=0 to t=1
    ts = torch.linspace(0, 1, n_steps, device=device)
    
    with torch.no_grad():
        for t in ts:
            v = model(t.expand(n_samples), x)
            # Simple Euler integration
            x = x + v * (1/n_steps)
    
    return x

if __name__ == "__main__":
    import torchvision.utils as vutils
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check available VRAM before loading/training
    if device == "cuda":
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    image_size = 256
    image_shape = (3, image_size, image_size)  # (C, H, W)
    image_dim = 3 * image_size * image_size
    
    # Check if model exists on disk
    model_path = "model.safetensors"
    if os.path.exists(model_path):
        print("Loading existing model from disk...")
        model = VectorField(image_dim=image_dim).to(device)
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found. Training new model...")
        model = train_flow_matching(device=device)
        # Save model
        model_state = model.state_dict()
        torch.save(model_state, model_path)
    
    # Generate samples
    os.makedirs("samples", exist_ok=True)
    for i in range(5):
        samples = sample(model, n_samples=16, device=device)
        vutils.save_image(samples, f"samples/samples_{i}.png", nrow=4, padding=2)
    print("Samples saved in 'samples' directory")