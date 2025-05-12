import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json
import shutil
from models.dnn_model import DNNSpeechEnhancer
from utils.data_loader import get_dataloaders

class SpeechEnhancementTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = DNNSpeechEnhancer(**config['model_config']).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training_config']['learning_rate'],
            weight_decay=config['training_config'].get('weight_decay', 1e-5)
        )
        self.criterion = nn.MSELoss()
        
        # Setup directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.abspath(f"saved_models/{self.timestamp}")
        os.makedirs(self.save_dir, exist_ok=True, mode=0o777)
        
        # Save config
        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Initialize logger
        self.writer = SummaryWriter(f"logs/{self.timestamp}")
        
        # Load data
        self.train_loader, self.val_loader = get_dataloaders(config)
        
        # Verify disk space
        self._check_disk_space()
    
    def _check_disk_space(self):
        """Check if there's enough disk space for saving checkpoints"""
        total, used, free = shutil.disk_usage(os.path.dirname(self.save_dir))
        min_space = 2 * 1024**3  # 2GB minimum
        if free < min_space:
            raise RuntimeError(f"Insufficient disk space. Need at least 2GB, only {free//(1024**2)}MB available")
    
    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        
        for noisy, clean in self.train_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(noisy)
            loss = self.criterion(outputs, clean)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
        return train_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                outputs = self.model(noisy)
                loss = self.criterion(outputs, clean)
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint with atomic write operation"""
        try:
            # Prepare state on CPU to reduce memory usage
            device_backup = next(self.model.parameters()).device
            self.model.to('cpu')
            
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss': val_loss,
                'config': self.config
            }
            
            # Save to temporary file first
            temp_path = os.path.join(self.save_dir, f"temp_checkpoint_{epoch}.pth")
            final_path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
            
            # Use alternative serialization if needed
            torch.save(state, temp_path, _use_new_zipfile_serialization=False)
            
            # Atomic rename
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)
            
            if is_best:
                best_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(state, best_path, _use_new_zipfile_serialization=False)
                
        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.remove(temp_path)
        finally:
            # Restore model to original device
            self.model.to(device_backup)
    
    def train(self):
        best_val_loss = float('inf')
        no_improve = 0
        
        for epoch in range(self.config['training_config']['epochs']):
            try:
                train_loss = self.train_epoch()
                val_loss = self.validate()
                
                # Logging
                self.writer.add_scalars('Loss', {
                    'train': train_loss,
                    'val': val_loss
                }, epoch)
                
                print(f"Epoch {epoch+1}/{self.config['training_config']['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
                
                # Save checkpoint only every 5 epochs or when it's the best
                if epoch % 5 == 0 or val_loss < best_val_loss:
                    self.save_checkpoint(epoch, val_loss, is_best=val_loss < best_val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve >= self.config['training_config'].get('patience', 5):
                    print(f"No improvement for {no_improve} epochs. Early stopping...")
                    break
                    
            except KeyboardInterrupt:
                print("\nTraining interrupted. Saving final checkpoint...")
                self.save_checkpoint(epoch, val_loss)
                break
            except Exception as e:
                print(f"Error during epoch {epoch}: {str(e)}")
                continue
                
        self.writer.close()

config = {
        "model_config": {
            "input_size": 16000,
            "hidden_sizes": [512, 512, 512],
            "output_size": 16000,
            "dropout": 0.2
        },
        "training_config": {
            "learning_rate": 0.001,
            "batch_size": 10,
            "epochs": 30,
            "patience": 10,
            "weight_decay": 1e-5,
            "checkpoint_interval": 10  # Save every 5 epochs
        },
        "data_config": {
            "base_dir": "data",
            "sample_rate": 16000
        }
    }

if __name__ == "__main__":
    
    
    # Validate data directory
    if not os.path.exists(config['data_config']['base_dir']):
        raise FileNotFoundError(f"Data directory not found: {config['data_config']['base_dir']}")
    
    try:
        trainer = SpeechEnhancementTrainer(config)
        trainer.train()
    except Exception as e:
        print(f"Fatal error during training: {str(e)}")