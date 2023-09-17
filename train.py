import torch
import gc

from sklearn.metrics import mean_absolute_error



def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        # compute the loss
        loss = criterion(outputs, labels.float().squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = outputs
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = mean_absolute_error(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        # backward & optimize
        loss.backward()
        optimizer.step()
        # break

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | mae {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = mean_absolute_error(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Mae', {'train': training_acc}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | mae: {:.2f}".format(epoch+1, training_loss, training_acc))