import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import SimpleCNN
from data_loading import FacialDepthDataset, transform
import os

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from a checkpoint file.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Save model and optimizer state to a checkpoint file.
    """
    print(f"Saving checkpoint to {filename}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def train():
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(in_channels=3, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint if available
    start_epoch = 0
    checkpoint_path = 'latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    # TensorBoard writer setup
    writer = SummaryWriter()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, depth_maps) in enumerate(train_loader):
            images = images.to(device)
            depth_maps = depth_maps.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, depth_maps)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item() * images.size(0)
            
            # Print progress every few batches
            if batch_idx % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Log loss to TensorBoard
        writer.add_scalar('Training Loss', epoch_loss, epoch + 1)
        
        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, epoch_loss, f'model_epoch_{epoch+1}.pth')
        
        # Save the latest checkpoint
        save_checkpoint(model, optimizer, epoch + 1, epoch_loss, 'latest_checkpoint.pth')

    writer.close()

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 5000  # Adjust based on training results
    batch_size = 25   # Start with 64 and increase if GPU memory allows
    save_interval = 100  # Save model every 5 epochs
    log_interval = 1  # Print progress every batch

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader setup
    dataset = FacialDepthDataset('without_background_images/', 'landmarks/', 'normalized_depth_maps/', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use a high number of workers to speed up data loading
    num_workers = 1  # Adjust based on CPU performance and system capabilities
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Start training
    train()
