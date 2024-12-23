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

#Like GeGLU but better gradient properties
class xATGLU(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        # GATE path | VALUE path
        self.proj = nn.Linear(input_dim, output_dim * 2, bias=bias)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.half_pi = torch.pi / 2
        self.inv_pi = 1 / torch.pi
        
    def forward(self, x):
        projected = self.proj(x)
        gate_path, value_path = projected.chunk(2, dim=-1)
        
        # Apply arctan gating with expanded range via learned alpha -- https://arxiv.org/pdf/2405.20768
        gate = (torch.arctan(gate_path) + self.half_pi) * self.inv_pi
        expanded_gate = gate * (1 + 2 * self.alpha) - self.alpha
        
        return expanded_gate * value_path  # g(x) × y

class ScaleBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.block = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim, bias=False),
            xATGLU(hidden_dim, hidden_dim // 4, bias=False),
            nn.Dropout(dropout_rate),
            xATGLU(hidden_dim // 4, hidden_dim, bias=False),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim, bias=False)
        )

    def forward(self, x, t):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Pre normalize to keep gradients from exploding, they giga die without this
        x_normed = self.norm(x_flat).view(x.shape)
        
        # Process normalized input wrt time
        t = t.view(batch_size, 1)
        x_t = torch.cat([x_flat, t], dim=-1)
        residual = self.block(x_t).view(x.shape)
        
        return x + residual

class MultiScaleVectorField(nn.Module):
    def __init__(self, input_shape, hidden_dims=[512, 256, 128], dropout_rate=0.1):
        super().__init__()
        self.sigma_min = 1e-3
        C, H, W = input_shape
        
        self.scales = nn.ModuleList([
            ScaleBlock(
                input_dim=C * (H // (2**i)) * (W // (2**i)),
                hidden_dim=h_dim,
                dropout_rate=dropout_rate
            )
            for i, h_dim in enumerate(hidden_dims)
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(hidden_dims)))
        
    def forward(self, x, t):
        results = []
        current = x
        target_size = x.shape[-2:]
        
        for i, scale in enumerate(self.scales):
            # Process at current scale
            v_t = scale(current, t)
            
            if v_t.shape[-2:] != target_size:
                v_t = F.interpolate(v_t, size=target_size, mode='nearest')
            
            results.append(v_t * self.scale_weights[i])
            
            if i < len(self.scales) - 1:
                current = F.avg_pool2d(current, 2)
        
        return sum(results)

def preload_dataset(image_size=256, device="cuda"):
    """Preload and cache the entire dataset in GPU memory"""
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("jiovine/pixel-art-nouns-2k", split="train")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to [-1, 1]
    ])
    
    # Process all images at once
    all_images = []
    for example in dataset:
        img_tensor = transform(example['image'])
        all_images.append(img_tensor)
        
    # Stack all images into a single tensor and move to GPU
    images_tensor = torch.stack(all_images).to(device)
    print(f"Dataset loaded: {images_tensor.shape} ({images_tensor.element_size() * images_tensor.nelement() / 1024/1024:.2f} MB)")
    
    return TensorDataset(images_tensor)

def train_matcha_flow(num_epochs=5000, initial_batch_sizes=[8, 16, 32, 64, 128], epoch_batch_drop_at=40, device="cuda"):
    dataset = preload_dataset(device=device)
    
    temp_loader = DataLoader(dataset, batch_size=initial_batch_sizes[0], shuffle=True)
    first_batch = next(iter(temp_loader))
    image_shape = first_batch[0].shape[1:] 
    image_dim = torch.prod(torch.tensor(image_shape)).item()
    
    model = MultiScaleVectorField(input_shape=image_shape).to(device)
    sigma_min = model.sigma_min
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=1e-3,
        warmup_steps=100
    )
    optimizer.train()

    current_batch_sizes = initial_batch_sizes.copy()
    
    next_drop_epoch = epoch_batch_drop_at
    interval_multiplier = 2
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Slowly decay batch size in a geometric style so e.g. 40 drop 120 drop etc..
        if epoch > 0 and epoch == next_drop_epoch and len(current_batch_sizes) > 1:
            current_batch_sizes.pop()  # Remove the largest batch size
            next_interval = epoch_batch_drop_at * interval_multiplier
            next_drop_epoch += next_interval
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
            
            # From flow matching for generative modeling: https://arxiv.org/abs/2210.02747
            # Sample t uniform in [0,1] 
            t = torch.rand(batch_size, 1, 1, 1, device=device)
            
            # Sample noise from standard normal
            x0 = torch.randn_like(x1)
            
            # Compute OT path interpolation (equation 22)
            sigma_t = 1 - (1 - sigma_min) * t
            mu_t = t * x1
            x_t = sigma_t * x0 + mu_t
            
            # Get vector field prediction
            v_t = model(x_t, t)
            
            # Compute target (equation 23)
            target = x1 - (1 - sigma_min) * x0
            
            # Compute loss
            loss = F.mse_loss(v_t, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Batch Size: {current_batch_size}, Average Loss: {avg_loss:.4f}, Learning Rate: {curr_lr:.6f}")
        
        if (epoch + 1) % 1 == 0:
            samples = sample(model, device=device)
            os.makedirs("samples", exist_ok=True)
            vutils.save_image(samples, f"samples/epoch_{epoch}.png", nrow=4, padding=2)
    
    return model

def sample(model, n_samples=16, n_steps=50, image_size=256, device="cuda", sigma_min=0.001):
    model.eval()
    
    # Euler integration for generative flow modeling, notably we're moving forward in time.
    # Start from pure noise (t=0)
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)

    # 0-1 space
    ts = torch.linspace(0, 1, n_steps, device=device)
    
    with torch.no_grad():
        for i in range(len(ts)):
            t = ts[i]
            t_input = t.repeat(n_samples).view(-1, 1)
            
            # Get vector field prediction (equation 21 in paper)
            v_t = model(x, t_input)
            
            # Compute time step (simple Euler integration)
            dt = 1/n_steps
            
            # Step forward in time (equation 1 in paper)
            x = x + v_t * dt
    
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = train_matcha_flow(device=device, initial_batch_sizes=[8, 16, 32, 64], epoch_batch_drop_at=100)
    
    for i in range(5):
        samples = sample(model, n_samples=16, device=device)
        vutils.save_image(samples, f"samples/epoch_{epoch}.jpg", nrow=4, padding=2, quality=90)
    
    print("Training complete! Samples saved in 'samples' directory")