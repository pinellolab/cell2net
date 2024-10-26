import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, dataloader: DataLoader, device, criterion, optimizer):
    model.train()

    train_loss = 0.0
    rna_true, rna_pred = [], []
    for data in dataloader:
        # get input features
        peak_seq = data["peak_seq"].to(device)
        peak_acc = data["peak_acc"].to(device)
        tf_exp = data["tf_exp"].to(device)
        covariates = data["covariates"].to(device)

        # get target gene expression
        target_exp = data["target_exp"].to(device)

        # get prediction
        pred_exp = model(peak_seq, peak_acc, tf_exp, covariates)
        loss = criterion(pred_exp.view(-1).float(), target_exp.view(-1).float())

        # optimize parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(dataloader)

        rna_true.append(target_exp.detach().cpu().view(-1))
        rna_pred.append(pred_exp.detach().cpu().view(-1))

    # convert log(lambda) to lambda
    rna_true = torch.concat(rna_true).exp().numpy()
    rna_pred = torch.concat(rna_pred).numpy()

    train_corr, _ = stats.spearmanr(rna_true, rna_pred)

    return train_loss, train_corr
