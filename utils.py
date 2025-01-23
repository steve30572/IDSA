import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset



def get_dataset(path, device, batch_size, transform=False):
    train_dataset = np.load(f'{path}/timeseries_train.npy')
    val_dataset = np.load(f'{path}/timeseries_val.npy')
    test_dataset = np.load(f'{path}/timeseries_test.npy')

    train_label = np.load(f'{path}/label_train.npy')
    val_label = np.load(f'{path}/label_val.npy')
    test_label = np.load(f'{path}/label_test.npy')

    if transform:
        train_dataset = np.transpose(train_dataset, (0, 2, 1))
        val_dataset = np.transpose(val_dataset, (0, 2, 1))
        test_dataset = np.transpose(test_dataset, (0, 2, 1))

    # Convert numpy arrays to PyTorch tensors
    train_dataset_tensor = torch.tensor(train_dataset, dtype=torch.float32, device=device)
    val_dataset_tensor = torch.tensor(val_dataset, dtype=torch.float32, device=device)
    test_dataset_tensor = torch.tensor(test_dataset, dtype=torch.float32, device=device)


    train_label_tensor = torch.tensor(train_label, dtype=torch.long, device=device)
    val_label_tensor = torch.tensor(val_label, dtype=torch.long,  device=device)
    test_label_tensor = torch.tensor(test_label, dtype=torch.long, device=device)

    # Create TensorDataset
    train_tensor_dataset = TensorDataset(train_dataset_tensor, train_label_tensor)
    val_tensor_dataset = TensorDataset(val_dataset_tensor, val_label_tensor)
    test_tensor_dataset = TensorDataset(test_dataset_tensor, test_label_tensor)

    # Create DataLoader
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, int(max(train_label).item())+1, train_dataset_tensor.shape[-1], train_dataset_tensor.shape[-2]