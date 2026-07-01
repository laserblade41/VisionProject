import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Fix: Import progress bar wrapper
from dataset_preparation import get_dataloaders
from dataset_preparation import SmartRestorationModel

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Use a standard batch size
    train_loader = get_dataloaders(batch_size=32, split="train")

    model = SmartRestorationModel()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    num_epochs = 3
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        # Fix: Wrap loader with tqdm to display real-time terminal progress
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.classifier(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Tracking statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
            # Update the progress bar description on the fly
            current_acc = (correct_preds.float() / total_samples).item()
            progress_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.2%}")

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds.float() / total_samples
        print(f"\n[Epoch {epoch+1} Finished] Global Loss: {epoch_loss:.4f} | Global Accuracy: {epoch_acc:.2%}\n")

    torch.save(model.state_dict(), "distortion_classifier.pth")
    print("Model weights saved successfully!")

if __name__ == "__main__":
    train_pipeline()