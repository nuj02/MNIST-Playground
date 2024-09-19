import torch

def validate(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    val_loss= 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            encoded, decoded = model(X)
            loss = loss_fn(decoded, X)

            val_loss += loss

    
    val_loss /= num_batches
    
    print(f"Validation Error: \n Avg loss: {val_loss:>8f}")

    return val_loss