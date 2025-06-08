# %%
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from model import Network
from data import corrupt_mnist
from tqdm import tqdm
import typer
from classif import classification

# %%
app= typer.Typer() #creating instance if the class

def train_validation(model, train_dataloaders, test_dataloaders, criterion, epochs, optimizer):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Training started...")
    
    for epoch in (range(epochs)):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_dataloaders, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
        
        for data_train, labels_train in train_bar:
            optimizer.zero_grad()
            output = model(data_train)
            loss = criterion(output, labels_train)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += labels_train.size(0)
            correct_train += (predicted == labels_train).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_train / total_train:.2f}%'
            })
        
        # Calculate average training metrics
        avg_train_loss = running_loss / len(train_dataloaders)
        train_accuracy = 100 * correct_train / total_train
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0
        
        test_bar = tqdm(test_dataloaders, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)
        
        with torch.no_grad():
            for data_test, labels_test in test_bar:
                output = model(data_test)
                loss = criterion(output, labels_test)
                running_loss_test += loss.item()
                
                # Calculate test accuracy
                _, predicted = torch.max(output.data, 1)
                total_test += labels_test.size(0)
                correct_test += (predicted == labels_test).sum().item()
                
                # Update progress bar
                test_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_test / total_test:.2f}%'
                })
        
            # Calculate average test metrics
            avg_test_loss = running_loss_test / len(test_dataloaders)
            test_accuracy = 100 * correct_test / total_test
            
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print('-' * 50)
    
    return train_losses, test_losses, train_accuracies, test_accuracies



# %%
# model = Network()
# train_dataloaders, test_dataloaders = corrupt_mnist()
# criterion = nn.CrossEntropyLoss()
# epochs= 20
# optimizer= optim.Adam(model.parameters(), lr = 0.001)
# train_losses, test_losses, train_accuracies, test_accuracies = train_validation(model, train_dataloaders, test_dataloaders, criterion, epochs, optimizer)


# %%
# Updated plotting function
def plotting(train_loss, test_loss, train_acc, test_acc, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_loss, label='Train Loss')
    ax1.plot(test_loss, label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_acc, label='Train Accuracy')
    ax2.plot(test_acc, label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

# %%
# plotting(train_losses, test_losses, train_accuracies, test_accuracies)

# %%
@app.command()
def training_model(
        lr: float = typer.Option(0.001, "--lr", help="Learning rate"),
        epochs: int = typer.Option(10, "--epochs", help="Number of epochs"),
        model_name: str = typer.Option("model.pth", "--model-name", help="Name to save the model"),
        plot_graph: bool = typer.Option(True, "--plot/--no-plot", help="Generate training plots"),
        save_path: str = typer.Option("graph.png","--graph-path", help="graph location"),
        batch_number : int = typer.Option(1,"--batch_number", help="batch_number"),
        image_path: str = typer.Option("graph2.png","--image_path", help="wrong_classification_image")):


        print(f"🚀 Starting training with:")
        print(f"   Learning rate: {lr}")
        print(f"   Epochs: {epochs}")
        print(f"   Model will be saved as: {model_name}")
        print(f"   Plot graphs: {plot_graph}")

        model = Network()
        train_dataloaders, test_dataloaders = corrupt_mnist()
        criterion = nn.CrossEntropyLoss()
        epochs= epochs
        optimizer= optim.Adam(model.parameters(), lr = lr)
        train_losses, test_losses, train_accuracies, test_accuracies = train_validation(model, train_dataloaders, test_dataloaders, criterion, epochs, optimizer)
    


        # saving name
        torch.save(model.state_dict(),model_name) 
        print(f"✅ Model saved successfully as: {model_name}")
        
        if plot_graph:
                print("📊 Generating training plots...")
                plotting(train_losses, test_losses, train_accuracies, test_accuracies,save_path)

        classification(batch_number,image_path)       
           
       
        
    
        
        

# %%
if __name__ == "__main__":
     app()


