import torch
from sklearn.metrics import accuracy_score,mean_absolute_error
import pickle
import numpy as np

def val_epoch(model, criterion, dataloader, device, epoch, logger, writer, phase='Train', exp_name = None):
    model.eval()
    losses = []
    all_label = []
    all_pred = []
    score_frag = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get the inputs and labels
            inputs_images, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs_images = []

            outputs_images.append(model(inputs_images))

                # if isinstance(outputs, list):
                #     outputs = outputs[0]
            outputs = torch.mean(torch.stack(outputs_images, dim=0), dim=0)
            if phase == 'Test':
                score_frag.append(outputs.data.cpu().numpy())
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = outputs
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
            if phase == 'Test':
                score = np.concatenate(score_frag)

        # Compute the average loss & accuracy
        validation_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        validation_mae = mean_absolute_error(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    if phase == 'Test':
        with open('./results/{}/results_epoch{:03d}_{}.pkl'.format(exp_name, epoch+1, validation_mae), 'wb') as f:
            score_dict = dict(zip(dataloader.dataset.sample_names, score))
            pickle.dump(score_dict, f)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Mae', {'validation': validation_mae}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | mae: {:.2f}%".format(epoch+1, validation_loss, validation_mae))
    return validation_loss