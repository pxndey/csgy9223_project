#!/usr/bin/env python
# coding: utf-8

# # Case 3: Adversarial Debiasing

# In[12]:


import pandas as pd
import joblib
x = pd.read_csv('data/X_caucasian_biased.csv')
y = pd.read_csv('data/y_caucasian_biased.csv')
df = pd.concat([x, y], axis=1)
df.head()


# In[ ]:


# data_prep.py (MODIFIED FOR BINARY SENSITIVE ATTRIBUTE)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
TARGET_COLUMN = 'diabetes'    
SENSITIVE_COLUMN = 'race'   

# Dummy Data Creation (Same as before)
def create_dummy_data():
    N_ROWS = 20000
    df = pd.DataFrame({
        f'feature_{i}': np.random.rand(N_ROWS) for i in range(5)
    })
    df['target'] = np.random.randint(0, 2, N_ROWS)
    df['feature_6'] = np.random.rand(N_ROWS)
    df[SENSITIVE_COLUMN] = np.random.choice(['Caucasian','Asian','AfricanAmerican','Hispanic','Other'], N_ROWS)
    return df.drop(columns=['feature_6']) 

try:
    data = df.copy()
except FileNotFoundError:
    print(f"File {DATA_FILE} not found. Creating dummy data...")
    data = create_dummy_data()

# --- 1. Encoding and Splitting ---

# 1.1 Binarize the sensitive attribute (S)
# Privileged: Caucasian (1), Unprivileged: All Others (0)
# This results in a binary target for the Adversary.
data['SENSITIVE_BINARIZED'] = data[SENSITIVE_COLUMN].apply(
    lambda x: 1 if x == 'Caucasian' else 0
)

# Set the new sensitive groups count
NUM_SENSITIVE_GROUPS = 2 # Now strictly binary!

# Separate features (X), primary target (Y), and sensitive target (S)
X = data.drop(columns=[TARGET_COLUMN, SENSITIVE_COLUMN, 'SENSITIVE_BINARIZED'])
Y = data[TARGET_COLUMN]
S = data['SENSITIVE_BINARIZED'] # Use the new binary target

# 1.2 Train/Test Split (80/20) - CRUCIAL STEP
X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
    X, Y, S, test_size=0.2, random_state=42, stratify=Y 
)

# Calculate class frequencies for the sensitive attribute S
s_counts = S_train.value_counts()
s_weights = 1.0 / s_counts
s_weights = s_weights / s_weights.sum() # Normalize weights

# Convert to a tensor for PyTorch
S_LOSS_WEIGHTS = torch.tensor(s_weights.sort_index().values, dtype=torch.float32)

# Add S_LOSS_WEIGHTS to your print statement so you know the weights:
print(f"S Loss Weights (0/1): {S_LOSS_WEIGHTS}")
# (If 0 is the minority, its weight should be much higher than 1's weight)

# --- 2. Normalization (Preventing Leakage) ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
filename = "scaler/adversarial_scaler.joblib"
joblib.dump(scaler, filename)
print(f"✅ Scaler successfully saved to {filename}")
# --- 3. Custom PyTorch Dataset (Same structure, different S data) ---

class ColumnarDataset(Dataset):
    def __init__(self, X_data, Y_data, S_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_data.values, dtype=torch.long)
        self.S = torch.tensor(S_data.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.S[idx]

# Create the final Dataset objects
train_dataset = ColumnarDataset(X_train_scaled, Y_train, S_train)
test_dataset = ColumnarDataset(X_test_scaled, Y_test, S_test)

# --- 4. DataLoader Setup ---

BATCH_SIZE = 128
INPUT_SIZE = X_train_scaled.shape[1] 
NUM_CLASSES = len(Y.unique())

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0
)

print(f"Input Size: {INPUT_SIZE}, Y Classes: {NUM_CLASSES}, S Groups: {NUM_SENSITIVE_GROUPS}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


# In[14]:


# adversarial_model.py (Adversary output MODIFIED to 2 logits, Architecture is FATTER)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model 1: Feature Extractor (F) (No change) ---
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim), 
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

# --- Model 2: Primary Classifier (C) (No change) ---
class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes) 
        
    def forward(self, z):
        return self.fc(z)

# --- Model 3: Adversary (A) (MODIFIED) ---
class Adversary(nn.Module):
    def __init__(self, feature_dim, num_sensitive_groups):
        super().__init__()
        self.net = nn.Sequential(
            # Fatter Architecture
            nn.Linear(feature_dim, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            # Output size is now num_sensitive_groups (which is 2)
            nn.Linear(64, num_sensitive_groups) 
        )

    def forward(self, z):
        return self.net(z)

# --- Container Model: Adversarial Pipeline (No change) ---
class AdversarialPipeline(nn.Module):
    def __init__(self, input_size, feature_dim, num_classes, num_sensitive_groups):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(input_size, feature_dim)
        self.classifier = Classifier(feature_dim, num_classes)
        # num_sensitive_groups is 2 here
        self.adversary = Adversary(feature_dim, num_sensitive_groups) 
        
    def forward(self, x):
        z = self.feature_extractor(x)
        y_pred = self.classifier(z)   
        s_pred = self.adversary(z)    
        
        return y_pred, s_pred


# In[15]:


# train_pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim# Import models from Step 2
from sklearn.metrics import accuracy_score

def train_and_validate(pipeline, train_loader, test_loader, device, epochs=10, feature_dim=32, lambda_adv=0.1):
    
    # 1. Optimizers Setup (Two separate ones)
    optimizer_main = optim.Adam(
        list(pipeline.feature_extractor.parameters()) + list(pipeline.classifier.parameters()),
        lr=0.001
    )
    optimizer_adv = optim.Adam(pipeline.adversary.parameters(), lr=0.001)

    # 2. Loss Functions
    criterion_Y = nn.CrossEntropyLoss()
    criterion_S = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    
    for epoch in range(1, epochs + 1):
        pipeline.train()
        total_y_loss, total_s_loss = 0, 0
        
        for x, y_true, s_true in train_loader:
            x, y_true, s_true = x.to(device), y_true.to(device), s_true.to(device)
            
            # --- PHASE 1: Train the ADVERSARY (A) to be accurate ---
            
            # Ensure only A is updated
            pipeline.feature_extractor.requires_grad_(False)
            pipeline.classifier.requires_grad_(False)
            pipeline.adversary.requires_grad_(True)
            
            optimizer_adv.zero_grad()
            
            # Forward pass: Crucial: .detach() z to prevent gradient flow to F/C
            z_detached = pipeline.feature_extractor(x).detach()
            s_pred_adv = pipeline.adversary(z_detached)
            
            loss_adv = criterion_S(s_pred_adv, s_true)
            loss_adv.backward()
            optimizer_adv.step()
            total_s_loss += loss_adv.item()

            # --- PHASE 2: Train F and C to be accurate on Y AND deceive A ---
            
            # Ensure F and C are updated, A is frozen
            pipeline.feature_extractor.requires_grad_(True)
            pipeline.classifier.requires_grad_(True)
            pipeline.adversary.requires_grad_(False)
            
            optimizer_main.zero_grad()

            # Full forward pass (no detach on Z)
            y_pred, s_pred_deceive = pipeline(x)
            
            # Main objective: Minimize Y loss
            loss_Y = criterion_Y(y_pred, y_true)
            
            # Adversarial objective: Maximize S loss (via negative sign)
            loss_S_deceive = criterion_S(s_pred_deceive, s_true)
            
            # Total Main Loss: L_Y - lambda * L_S_deceive (Minimize this total)
            total_main_loss = loss_Y - lambda_adv * loss_S_deceive
            
            total_main_loss.backward()
            optimizer_main.step()
            total_y_loss += loss_Y.item()

        # --- Validation after each epoch ---
        val_acc, val_s_acc = evaluate_model(pipeline, test_loader, device)
        
        print(f"Epoch {epoch:02d} | Avg Y Loss: {total_y_loss/len(train_loader):.4f} | Avg Adv Loss: {total_s_loss/len(train_loader):.4f}")
        print(f"        | Val Y Acc: {val_acc*100:.2f}% | Val S Acc (Adversary success): {val_s_acc*100:.2f}%")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Save the best model based on primary task accuracy
            save_checkpoint(pipeline, optimizer_main, optimizer_adv, epoch, best_val_accuracy)

def evaluate_model(pipeline, data_loader, device):
    pipeline.eval()
    y_true_all, y_pred_all = [], []
    s_true_all, s_pred_all = [], []
    
    with torch.no_grad():
        for x, y_true, s_true in data_loader:
            x = x.to(device)
            y_pred_logits, s_pred_logits = pipeline(x)
            
            # Primary Task (Y) Accuracy
            y_preds = torch.argmax(y_pred_logits, dim=1)
            y_pred_all.extend(y_preds.cpu().numpy())
            y_true_all.extend(y_true.cpu().numpy())
            
            # Adversary Task (S) Accuracy
            s_preds = torch.argmax(s_pred_logits, dim=1)
            s_pred_all.extend(s_preds.cpu().numpy())
            s_true_all.extend(s_true.cpu().numpy())
            
    y_acc = accuracy_score(y_true_all, y_pred_all)
    s_acc = accuracy_score(s_true_all, s_pred_all)
    return y_acc, s_acc

def save_checkpoint(pipeline, optimizer_main, optimizer_adv, epoch, val_accuracy):
    CHECKPOINT_PATH = "best_adversarial_pipeline_checkpoint.pth"
    print(f"--- Saving best model (Acc: {val_accuracy*100:.2f}%) ---")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': pipeline.state_dict(),
        'optimizer_main_state_dict': optimizer_main.state_dict(),
        'optimizer_adv_state_dict': optimizer_adv.state_dict(),
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


# In[16]:


# train_pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Helper to freeze/unfreeze entire modules
def toggle_grad(module, enable):
    for param in module.parameters():
        param.requires_grad = enable

        
def train_and_validate(pipeline, train_loader, test_loader, device, epochs=10, feature_dim=32, lambda_adv=0.1, s_loss_weights=None):
    
    # 1. Optimizers Setup (Two separate ones)
    optimizer_main = optim.Adam(
        list(pipeline.feature_extractor.parameters()) + list(pipeline.classifier.parameters()),
        lr=0.001
    )
    optimizer_adv = optim.Adam(pipeline.adversary.parameters(), lr=0.001)

    # 2. Loss Functions
    criterion_Y = nn.CrossEntropyLoss()
    criterion_S = nn.CrossEntropyLoss(weight=s_loss_weights) # <-- USE WEIGHTS HERE

    best_val_accuracy = 0.0
    
    for epoch in range(1, epochs + 1):
        pipeline.train()
        total_y_loss, total_s_loss = 0, 0
        
        for x, y_true, s_true in train_loader:
            x, y_true, s_true = x.to(device), y_true.to(device), s_true.to(device)
            
            # =====================================================================
            # --- PHASE 1: Train the ADVERSARY (A) to be accurate ---
            # Objective: Minimize Loss_S
            # =====================================================================
            
            # 1. Ensure only Adversary parameters are active
            toggle_grad(pipeline.feature_extractor, False)
            toggle_grad(pipeline.classifier, False)
            toggle_grad(pipeline.adversary, True) 
            
            optimizer_adv.zero_grad()
            
            # Forward pass for A's loss: F must be run inside torch.no_grad() 
            # to ensure Z has no history, guaranteeing that F is not updated.
            with torch.no_grad():
                z = pipeline.feature_extractor(x)
            
            # Adversary uses the frozen feature Z
            s_pred_adv = pipeline.adversary(z) 
            
            # Calculate loss for A
            loss_adv = criterion_S(s_pred_adv, s_true)
            loss_adv.backward()
            optimizer_adv.step()
            
            total_s_loss += loss_adv.item() # <-- NOW THIS SHOULD BE NON-ZERO

            # =====================================================================
            # --- PHASE 2: Train F and C to deceive A ---
            # Objective: Minimize Loss_Y - Lambda * Loss_S
            # =====================================================================
            
            # 2. Ensure F and C are active, A is frozen
            toggle_grad(pipeline.feature_extractor, True)
            toggle_grad(pipeline.classifier, True)
            toggle_grad(pipeline.adversary, False)
                
            optimizer_main.zero_grad()

            # Full forward pass for main loss calculation (Z history is tracked now)
            y_pred, s_pred_deceive = pipeline(x) 
            
            loss_Y = criterion_Y(y_pred, y_true)
            loss_S_deceive = criterion_S(s_pred_deceive, s_true)
            
            total_main_loss = loss_Y - LAMBDA_ADV * loss_S_deceive 
            
            total_main_loss.backward()
            optimizer_main.step()
            total_y_loss += loss_Y.item()

        # --- Validation after each epoch ---
        val_acc, val_s_acc = evaluate_model(pipeline, test_loader, device)
        
        print(f"Epoch {epoch:02d} | Avg Y Loss: {total_y_loss/len(train_loader):.4f} | Avg Adv Loss: {total_s_loss/len(train_loader):.4f}")
        print(f"        | Val Y Acc: {val_acc*100:.2f}% | Val S Acc (Adversary success): {val_s_acc*100:.4f}%")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Save the best model based on primary task accuracy
            save_checkpoint(pipeline, optimizer_main, optimizer_adv, epoch, best_val_accuracy)

def evaluate_model(pipeline, data_loader, device):
    pipeline.eval()
    y_true_all, y_pred_all = [], []
    s_true_all, s_pred_all = [], []
    
    with torch.no_grad():
        for x, y_true, s_true in data_loader:
            x = x.to(device)
            y_pred_logits, s_pred_logits = pipeline(x)
            
            # Primary Task (Y) Accuracy
            y_preds = torch.argmax(y_pred_logits, dim=1)
            y_pred_all.extend(y_preds.cpu().numpy())
            y_true_all.extend(y_true.cpu().numpy())
            
            # Adversary Task (S) Accuracy
            s_preds = torch.argmax(s_pred_logits, dim=1)
            s_pred_all.extend(s_preds.cpu().numpy())
            s_true_all.extend(s_true.cpu().numpy())
            
    y_acc = accuracy_score(y_true_all, y_pred_all)
    s_acc = accuracy_score(s_true_all, s_pred_all)
    return y_acc, s_acc

def save_checkpoint(pipeline, optimizer_main, optimizer_adv, epoch, val_accuracy):
    CHECKPOINT_PATH = "best_adversarial_pipeline_checkpoint.pth"
    print(f"--- Saving best model (Acc: {val_accuracy*100:.2f}%) ---")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': pipeline.state_dict(),
        'optimizer_main_state_dict': optimizer_main.state_dict(),
        'optimizer_adv_state_dict': optimizer_adv.state_dict(),
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


# In[17]:


# main_script.py
if __name__ == "__main__":
    import torch
    import os

    # --- Configuration ---
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    FEATURE_DIM = 32 # Size of the shared feature vector Z
    EPOCHS = 25
    LAMBDA_ADV = 10 # Hyperparameter: controls the strength of the debiasing pressure (0.1 to 1.0)

    # --- Initialize Model ---
    pipeline = AdversarialPipeline(
        input_size=INPUT_SIZE, 
        feature_dim=FEATURE_DIM, 
        num_classes=NUM_CLASSES, 
        num_sensitive_groups=NUM_SENSITIVE_GROUPS
    ).to(DEVICE)

    print(f"--- Model Initialized on {DEVICE} ---")
    print(f"Total Parameters: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,}")
    print(f"Starting Training for {EPOCHS} epochs with Lambda={LAMBDA_ADV}")

    # --- Start Training ---
    train_and_validate(
        pipeline, 
        train_loader, 
        test_loader, 
        DEVICE, 
        EPOCHS, 
        FEATURE_DIM, 
        LAMBDA_ADV,
        S_LOSS_WEIGHTS.to(DEVICE) # <-- PASS WEIGHTS TO TRAIN FUNCTION
    )

    print("\n--- Training Finished ---")
    print("Best model checkpoint saved to 'best_adversarial_pipeline_checkpoint.pth'")

