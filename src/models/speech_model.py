# src/models/speech_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..preprocessing.audio_processor import AudioProcessor
import logging
import json


logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out(context)

class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with depthwise separable convolutions"""
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.depthwise = nn.Conv1d(
            input_channels, input_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size//2 * dilation,
            groups=input_channels, dilation=dilation
        )
        self.pointwise = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(output_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(output_channels)
        ) if input_channels != output_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.bn2(out)

        identity = self.skip(identity)
        out += identity
        out = self.relu(out)

        return out

class AttentivePooling(nn.Module):
    """
    Attentive pooling mechanism for better temporal aggregation of audio features
    """
    def __init__(self, in_dim):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        """
        Apply attentive pooling over the time dimension

        Args:
            x: Input tensor of shape [batch_size, time_steps, features]

        Returns:
            Pooled tensor of shape [batch_size, features]
        """
        # Calculate attention weights
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention
        weighted = x * attn_weights

        # Sum over time dimension
        pooled = weighted.sum(dim=1)

        return pooled

# class TwiSpeechModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
#         """
#         Speech recognition model for Twi language using BiLSTM

#         Args:
#             input_dim: Input dimension (num features)
#             hidden_dim: Hidden dimension
#             num_classes: Number of output classes
#             num_layers: Number of LSTM layers
#             dropout: Dropout probability
#         """
#         super(TwiSpeechModel, self).__init__()

#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0,
#             bidirectional=True
#         )

#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#             nn.Softmax(dim=1)
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         """
#         Forward pass

#         Args:
#             x: Input tensor of shape (batch_size, seq_len, input_dim)

#         Returns:
#             Output tensor of shape (batch_size, num_classes)
#         """
#         # BiLSTM layer
#         lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)

#         # Attention mechanism
#         attn_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
#         context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_dim*2)

#         # Fully connected layers
#         output = self.fc(context)  # (batch_size, num_classes)

#         return output

class ImprovedTwiSpeechModel(nn.Module):
    """
    Enhanced speech recognition model with Transformer attention and CNN layers.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, num_heads=4):
        super(ImprovedTwiSpeechModel, self).__init__()

        # Store input_dim for reference
        self.input_dim = input_dim
        print(f"Creating ImprovedTwiSpeechModel with input_dim={input_dim}")

        # Depthwise separable convolutions - properly implemented
        # For depthwise conv, we use groups=input_dim but keep in_channels=out_channels=input_dim
        # Then use a pointwise (1x1) conv to change the number of channels
        self.conv1 = nn.Sequential(
            # Depthwise convolution - keeps same number of channels
            nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2, groups=input_dim),
            # Pointwise convolution - changes number of channels
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),  # Use BatchNorm instead of LayerNorm for 1D convs
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            # Depthwise
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64),
            # Pointwise
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            # Depthwise
            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=128),
            # Pointwise
            nn.Conv1d(128, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # BiLSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Transformer attention
        self.attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)

        # Classification layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout / 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape check
        if len(x.shape) == 3:
            # If input is [batch, seq_len, features], transpose to [batch, features, seq_len]
            if x.shape[1] != self.input_dim and x.shape[2] == self.input_dim:
                x = x.transpose(1, 2)
            # Could add more checks here if needed

        # CNN blocks
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # BiLSTM with residual connection
        x3 = x3.transpose(1, 2)  # Convert back to batch_size, seq_len, hidden_dim
        lstm_out, _ = self.lstm(x3)
        lstm_out = lstm_out + self.residual(lstm_out)  # Residual connection

        # Transformer attention
        attn_out = self.attention(lstm_out)

        # Classification - use global average pooling
        x = F.relu(self.fc1(attn_out.mean(dim=1)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class EnhancedTwiSpeechModel(nn.Module):
    """
    Advanced speech recognition model with improved architecture
    including squeeze-and-excitation blocks, attention mechanisms, and residual connections.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, num_heads=4):
        super(EnhancedTwiSpeechModel, self).__init__()

        self.input_dim = input_dim
        print(f"Creating EnhancedTwiSpeechModel with input_dim={input_dim}")

        # Squeeze-and-Excitation block for channel attention
        self.se_block = SqueezeExciteBlock(input_dim)

        # Initial convolution with batch normalization
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )

        # Depthwise separable convolutions with residual connections
        self.conv_blocks = nn.ModuleList([
            ResidualBlock(
                input_channels=input_dim if i==0 else 64*(2**min(i-1, 2)),
                output_channels=64*(2**min(i, 2)),
                kernel_size=5 if i==0 else 3,
                stride=1,
                dilation=i+1
            )
            for i in range(3)
        ])

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(50)

        # Add a dimension matching layer - THIS IS THE FIX
        # Calculate the output dimension from the last conv block (256 for 3 blocks)
        conv_output_dim = 64 * (2**min(2, 2))  # 256 for 3 blocks

        # Add a linear layer to match dimensions between conv and GRU
        self.dim_matching = nn.Linear(conv_output_dim, hidden_dim)

        # BiGRU instead of LSTM for faster training with similar performance
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)

        # Two-stage classification head with skip connection
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

        # Final classification layer
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Improved weight initialization for faster convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape check and adjustment
        if len(x.shape) == 3:
            if x.shape[1] != self.input_dim and x.shape[2] == self.input_dim:
                x = x.transpose(1, 2)

        # Apply channel attention
        x = self.se_block(x)

        # Initial convolution
        x = self.initial_conv(x)

        # Apply convolutional blocks with residual connections
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Temporal pooling to standardize sequence length
        x = self.temporal_pool(x)

        # Prepare for RNN (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        x = self.dim_matching(x)  # This will transform from whatever size to hidden_dim

        # Apply bidirectional GRU
        gru_out, _ = self.gru(x)

        # Apply attention
        attn_out = self.attention(gru_out)

        # Global average and max pooling, then concatenate
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        pooled = avg_pool + max_pool  # Element-wise addition works as a lightweight attention

        # Classification head
        features = self.classification_head(pooled)
        output = self.fc_out(features)

        return output


class Trainer:
    def __init__(self, model, device, learning_rate=0.001, model_dir="data/models"):
        """
        Trainer for Twi speech model

        Args:
            model: PyTorch model
            device: Device to use (cpu or cuda)
            learning_rate: Learning rate
            model_dir: Directory to save models
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.model_dir = model_dir

        os.makedirs(model_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training"):
            print(f"Input shape before device: {inputs.shape}")
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            print(f"Input shape: {inputs.shape}")
            print(f"Conv1D expects in_channels: {self.model.conv1.in_channels}")

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=10, early_stopping_patience=5):
        """
        Train model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(self.model_dir, "best_model.pt"))
                print("Best model saved!")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(self.model_dir, f"model_epoch_{epoch+1}.pt"))

        # Plot training history
        self.plot_history()

        return self.history

    def save_model(self, path):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, path):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

    def plot_history(self):
            """Plot training history"""
            plt.figure(figsize=(12, 4))

            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')

            # Plot accuracy
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_acc'], label='Train Accuracy')
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Curves')

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
            plt.close()


class ImprovedTrainer:
    def __init__(self, model, device, learning_rate=0.001, model_dir="data/models"):
        self.model = model
        self.device = device
        self.model.to(device)

        # Use AdamW instead of Adam - better weight decay handling
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # L2 regularization
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Use weighted loss for imbalanced classes (will be set during training)
        self.criterion = nn.CrossEntropyLoss()
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # For mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }

    def compute_class_weights(self, train_loader):
        """Compute class weights for imbalanced datasets"""
        # Count labels
        label_counts = {}
        total_samples = 0

        for _, labels in train_loader:
            for label in labels:
                label_idx = label.item()
                label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
                total_samples += 1

        # Compute weights (inversely proportional to frequency)
        num_classes = len(label_counts)
        weights = torch.zeros(num_classes)

        for label_idx, count in label_counts.items():
            weights[label_idx] = total_samples / (count * num_classes)

        return weights.to(self.device)

    def train_epoch(self, train_loader):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed precision training if using GPU
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Scale gradients and optimize
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)

        # Update learning rate scheduler
        self.scheduler.step(epoch_loss)

        return epoch_loss, epoch_acc, all_preds, all_targets

    def train(self, train_loader, val_loader, num_epochs=10, early_stopping_patience=5, class_weighting=True):
        """
        Train model with early stopping and learning rate scheduling
        """
        # Set up class weights if needed
        if class_weighting:
            print("Computing class weights for balanced learning...")
            class_weights = self.compute_class_weights(train_loader)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Class weights: {class_weights}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(self.model_dir, "best_model.pt"))
                print("Best model saved!")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(self.model_dir, f"model_epoch_{epoch+1}.pt"))

            print("")

        # Plot training history
        self.plot_history()

        return self.history

    def save_model(self, path):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, path):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint['history']

    def plot_history(self):
        """Plot enhanced training history"""
        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves')

        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        # Add a text summary
        plt.subplot(2, 2, 4)
        plt.axis('off')

        final_train_loss = self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A'
        final_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else 'N/A'
        final_train_acc = self.history['train_acc'][-1] if self.history['train_acc'] else 'N/A'
        final_val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 'N/A'

        summary = (
            f"Training Summary:\n\n"
            f"Final Training Loss: {final_train_loss:.4f}\n"
            f"Final Validation Loss: {final_val_loss:.4f}\n"
            f"Final Training Accuracy: {final_train_acc:.2f}%\n"
            f"Final Validation Accuracy: {final_val_acc:.2f}%\n"
            f"Epochs: {len(self.history['train_loss'])}"
        )

        plt.text(0.1, 0.5, summary, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.close()



class AdvancedTrainer:
    """
    Advanced trainer with mixed precision, learning rate scheduling,
    gradient clipping, and other modern training techniques.
    """
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.model.to(device)
        self.config = config

        # Learning rate and optimizer
        self.base_lr = config.get('learning_rate', 0.001)

        # Use AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )

        # Mixed precision training
        self.use_amp = device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Learning rate scheduler - One Cycle Policy
        steps_per_epoch = config.get('steps_per_epoch', 100)
        epochs = config.get('num_epochs', 50)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.base_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,  # Spend 30% of time warming up
            div_factor=10,  # Initial LR is max_lr/10
            final_div_factor=100  # Final LR is max_lr/1000
        )

        # Criterion with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Gradient clipping value
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)

        # Early stopping parameters
        self.patience = config.get('early_stopping_patience', 10)
        self.min_delta = config.get('early_stopping_min_delta', 0.001)

        # Model directory
        self.model_dir = config.get('model_dir', 'data/models')
        os.makedirs(self.model_dir, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with mixed precision and gradient accumulation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

        # Reset gradients at start of epoch
        self.optimizer.zero_grad()

        # Track metrics
        batch_losses = []
        batch_accuracies = []

        # Gradient accumulation steps
        accumulation_steps = self.config.get('accumulation_steps', 1)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / accumulation_steps

                # Scale gradients and optimize
                self.scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    # Step optimizer and update scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # Update LR scheduler
                    self.scheduler.step()
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    # Step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update LR scheduler
                    self.scheduler.step()

            # Calculate accuracy and accumulate loss
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            batch_loss = loss.item() * accumulation_steps
            batch_acc = 100 * predicted.eq(targets).sum().item() / targets.size(0)
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_acc)

            progress_bar.set_postfix({
                'loss': f"{np.mean(batch_losses[-10:]):.4f}",
                'acc': f"{np.mean(batch_accuracies[-10:]):.2f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Update history
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

        return epoch_loss, epoch_acc


class IntentClassifier:
    """Optimized intent classifier with support for multiple model types."""
    def __init__(self, model_path, device, processor=None, label_map_path=None, model_type='standard'):
        self.device = device
        self.processor = processor if processor else AudioProcessor()
        self.model_type = model_type

        # Load label map
        if label_map_path and os.path.exists(label_map_path):
            self.label_map = np.load(label_map_path, allow_pickle=True).item()
            num_classes = len(self.label_map)
        else:
            self.label_map = None
            num_classes = 20  # Default fallback

        # Determine input dimension
        input_dim = self._determine_input_dim(model_path)
        logger.info(f"Using input dimension: {input_dim}")

        # Initialize appropriate model based on type
        if model_type == 'enhanced':
            logger.info("Initializing enhanced speech model...")
            self.model = EnhancedTwiSpeechModel(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=num_classes
            )
        else:
            logger.info("Initializing standard speech model...")
            self.model = ImprovedTwiSpeechModel(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=num_classes
            )

        # Load model weights
        self._load_model_weights(model_path)

        # Set model to evaluation mode
        self.model.eval()

    def _determine_input_dim(self, model_path):
        """Determine input dimension from model or config"""
        # Try to get from model info file
        model_dir = os.path.dirname(model_path)
        model_info_path = os.path.join(model_dir, 'model_info.json')

        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    return model_info.get('input_dim', 94)
            except Exception as e:
                logger.warning(f"Could not read model_info.json: {e}")

        # Try to extract from checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                return checkpoint['config'].get('input_dim', 94)
        except Exception as e:
            logger.warning(f"Could not extract input_dim from checkpoint: {e}")

        # Fallback to default
        return 94

    def _load_model_weights(self, model_path):
        """Load model weights from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict itself is the state dict
                    self.model.load_state_dict(checkpoint)
            else:
                logger.warning(f"Unexpected checkpoint format, attempting to load directly")
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise

    def classify(self, audio_path):
        """Classify audio and return intent and confidence"""
        # Preprocess audio to get features
        features = self.processor.preprocess(audio_path)

        # Ensure features have the right dimensions for the model
        input_channels = self.model.input_dim if hasattr(self.model, 'input_dim') else 94
        if features.shape[0] != input_channels:
            logger.warning(f"Feature dimension mismatch. Model expects {input_channels}, got {features.shape[0]}")

            # Adjust dimensions
            if features.shape[0] > input_channels:
                features = features[:input_channels, :]
            else:
                padding = np.zeros((input_channels - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))

        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        # Map index to intent label
        if self.label_map:
            idx_to_label = {v: k for k, v in self.label_map.items()}
            intent = idx_to_label.get(predicted_idx.item(), f"unknown_{predicted_idx.item()}")
        else:
            intent = f"intent_{predicted_idx.item()}"

        return intent, confidence.item()
