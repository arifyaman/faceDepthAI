import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from model import SimpleLandmarkOnlyCNN
from data_loading import FacialDepthDatasetWithLandmarks

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_checkpoint(epoch, model, optimizer, loss, filename):
    """ Save the model checkpoint """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    logging.info(f'Checkpoint saved to {filename}')

def load_checkpoint(filename, model, optimizer):
    """ Load the model checkpoint """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f'Checkpoint loaded from {filename}')
        return epoch, loss
    else:
        logging.info(f'No checkpoint found at {filename}')
        return 0, None

def train():
    # Hyperparameters
    num_epochs = 1000
    batch_size = 50
    save_interval = 100
    num_workers = 1  # Adjust based on system capabilities
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader setup
    dataset = FacialDepthDatasetWithLandmarks('train_data/landmarks/', 'train_data/only_face_depth_maps/')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleLandmarkOnlyCNN(landmark_dim=468, out_channels=1, depth_map_size=(85, 85)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint if available
    start_epoch, _ = load_checkpoint('checkpoint.pth', model, optimizer)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (landmarks, depth_maps) in enumerate(train_loader):
            landmarks = landmarks.to(device)
            depth_maps = depth_maps.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(landmarks)
            
            # Calculate loss
            loss = criterion(outputs, depth_maps)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * landmarks.size(0)
            running_loss += batch_loss
            
            # Log batch loss
            logging.info(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Batch Loss: {batch_loss:.4f}')
        
        # Calculate and log epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Save model checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0:
            epoch_checkpoint_filename = f'model_epoch_{epoch + 1}.pth'
            save_checkpoint(epoch, model, optimizer, epoch_loss, filename=epoch_checkpoint_filename)
            logging.info(f'Model checkpoint saved after epoch {epoch + 1}')
        
        # Always save the latest checkpoint
        save_checkpoint(epoch, model, optimizer, epoch_loss, filename='checkpoint.pth')

if __name__ == '__main__':
    train()
