def train(dataloader, model, loss_fn, optimizer, device, scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        encoded, decoded = model(X) 
        loss = loss_fn(X, decoded) #use decoder output to calculate loss with encoder input

        #backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss 
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    scheduler.step()

    train_loss /= num_batches

    print(f"Training Error: \n Avg loss: {train_loss:>8f}")

    return train_loss