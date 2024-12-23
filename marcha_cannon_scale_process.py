import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from schedulefree import AdamWScheduleFree
from torch.nn.utils.parametrizations import weight_norm
import os

class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        # Keep the shared projection
        self.proj = nn.Linear(input_dim, output_dim * 2, bias=bias)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        self.norm = nn.LayerNorm(output_dim * 2)

    def forward(self, x):
        # Project and normalize
        projected = self.proj(x)
        normalized = self.norm(projected)
        
        # Split the normalized output
        gelu_path, gate = normalized.chunk(2, dim=-1)
        
        # Apply GELU to the main path and multiply with gate
        return F.gelu(gelu_path) * gate

class ScaleBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim, bias=False),
            GeGLU(hidden_dim, hidden_dim // 4),
            nn.Dropout(dropout_rate),
            GeGLU(hidden_dim // 4, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim, bias=False)
        )
    
    def forward(self, x, t):
        batch_size = x.shape[0]
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, -1)
        x_t = torch.cat([x_flat, t.view(-1, 1)], dim=-1)
        return self.block(x_t).view(x.shape)

class MultiScaleVectorField(nn.Module):
    def __init__(self, input_shape, hidden_dims=[512, 256, 128, 64], dropout_rate=0.1):
        super().__init__()
        self.sigma_min = 1e-3
        C, H, W = input_shape
        
        # Create scale-specific blocks
        self.scales = nn.ModuleList([
            ScaleBlock(
                input_dim=C * (H // (2**(len(hidden_dims)-1-i))) * (W // (2**(len(hidden_dims)-1-i))),
                hidden_dim=h_dim,
                dropout_rate=dropout_rate
            )
            for i, h_dim in enumerate(hidden_dims)
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(hidden_dims)))
        
        # Optional: Scale-specific processing
        self.scale_process = nn.ModuleList([
            nn.Conv2d(C, C, 3, padding=1)
            for _ in range(len(hidden_dims))
        ])
        
    def forward(self, x, t):
        results = []
        # Start with maximally downsampled version
        pool_size = 8  # This will downsample 256->32
        current = F.avg_pool2d(x, pool_size)
        
        for i, (scale, process) in enumerate(zip(self.scales, self.scale_process)):
            # Process at current scale
            v_t = scale(current, t)
            v_t = process(v_t)
            
            # Upsample to original size for final output
            if v_t.shape != x.shape:
                v_t = F.interpolate(v_t, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
            results.append(v_t * self.scale_weights[i].view(1, 1, 1, 1))
            
            # Upsample for next scale if not at final scale
            if i < len(self.scales) - 1:
                current = F.interpolate(current, scale_factor=2, mode='bilinear', align_corners=False)
                
        return sum(results)


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
        img_tensor = transform(example['image'].convert('RGB'))
        all_images.append(img_tensor)

    # Stack all images into a single tensor and move to GPU
    images_tensor = torch.stack(all_images).to(device)
    print(f"Dataset loaded: {images_tensor.shape} ({images_tensor.element_size() * images_tensor.nelement() / 1024/1024:.2f} MB)")
    return TensorDataset(images_tensor)

def train_matcha_flow(num_epochs=5000, initial_batch_sizes=[8, 16, 32, 64, 128], epoch_batch_drop_at=40, device="cuda"):
    # Preload the dataset
    dataset = preload_dataset(device=device)
    
    # Initialize model and move to device
    temp_loader = DataLoader(dataset, batch_size=initial_batch_sizes[0], shuffle=True)
    first_batch = next(iter(temp_loader))
    image_shape = first_batch[0].shape[1:] 
    image_dim = torch.prod(torch.tensor(image_shape)).item()
    
    model = MultiScaleVectorField(input_shape=image_shape).to(device)
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=1e-3,
        warmup_steps=100
    )
    optimizer.train()

    # Create a mutable list of batch sizes that we'll modify during training
    current_batch_sizes = initial_batch_sizes.copy()
    
    # Keep track of next drop epoch
    next_drop_epoch = epoch_batch_drop_at
    # Keep track of multiplier for interval calculation
    interval_multiplier = 2
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Remove largest batch size at increasing intervals
        if epoch > 0 and epoch == next_drop_epoch and len(current_batch_sizes) > 1:
            current_batch_sizes.pop()  # Remove the largest batch size
            # Calculate next interval: base_interval * current_multiplier
            next_interval = epoch_batch_drop_at * interval_multiplier
            # Update next drop epoch
            next_drop_epoch += next_interval
            # Increment multiplier for next time
            interval_multiplier += 1
            
            print(f"\nEpoch {epoch}: Reducing batch size to {current_batch_sizes[-1]}")
            print(f"Next drop will occur at epoch {next_drop_epoch} (interval: {next_interval})")
        
        # Always use the largest remaining batch size
        current_batch_size = current_batch_sizes[-1]
        dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)
        
        curr_lr = optimizer.param_groups[0]['lr']
        
        for batch in dataloader:
            x1 = batch[0]
            batch_size = x1.shape[0]
            
            u = torch.randn(batch_size, device=device)  # N(0,1)
            t = torch.sigmoid(u)  # Maps to (0,1)

            s = 1.0  # scale parameter
            m = 0.0  # location parameter
            weight = (1 / (s * 2.506627216)) * (1 / (t * (1 - t))) * torch.exp(-(u - m)**2 / (2 * s**2))
            weight = weight.reshape(-1, 1, 1, 1)
            
            x0 = torch.randn_like(x1)
            x_t = (1 - (1 - model.sigma_min) * t.reshape(-1, 1, 1, 1)) * x0 + t.reshape(-1, 1, 1, 1) * x1
            v_t = model(x_t, t)
            target = x1 - (1 - model.sigma_min) * x0
            
            # Apply the weighting to MSE loss
            loss = (weight * (v_t - target).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Batch Size: {current_batch_size}, Average Loss: {avg_loss:.4f}, Learning Rate: {curr_lr:.6f}")
        
        if (epoch + 1) % 1 == 0:
            samples = sample_matcha(model, device=device)
            os.makedirs("samples", exist_ok=True)
            vutils.save_image(samples, f"samples/epoch_{epoch}.png", nrow=4, padding=2)
    
    return model

def sample_matcha(model, n_samples=16, n_steps=10, image_size=256, device="cuda"):
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1 / n_steps
    
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.ones(n_samples, device=device) * (1 - i * dt)  # Reverse time direction
            v = model(x, t)
            x = x + v * dt
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Train model
    model = train_matcha_flow(device=device, initial_batch_sizes=[8, 16, 32], epoch_batch_drop_at=600)
    
    # Generate final samples
    for i in range(5):
        samples = sample_matcha(model, n_samples=16, device=device)
        vutils.save_image(samples, f"samples/epoch_{epoch}.jpg", nrow=4, padding=2, quality=90)
    
    print("Training complete! Samples saved in 'samples' directory")