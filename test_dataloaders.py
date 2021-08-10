from data.dataloaders.bar_dataset import *

is_short = False
num_bars = 16
if is_short:
    batch_size = 10
else:
    batch_size = 128
bar_dataset = FolkNBarDataset(dataset_type='train', is_short=is_short, num_bars=num_bars)
(train_dataloader,
 val_dataloader,
 test_dataloader) = bar_dataset.data_loaders(
    batch_size=batch_size,
    split=(0.7, 0.2)
)
print('Num Train Batches: ', len(train_dataloader))
print('Num Valid Batches: ', len(val_dataloader))
print('Num Test Batches: ', len(test_dataloader))

bar_dataset = FolkNBarDataset(dataset_type='test', is_short=is_short, num_bars=num_bars)
(train_dataloader,
 val_dataloader,
 test_dataloader) = bar_dataset.data_loaders(
    batch_size=batch_size,
    split=(0.7, 0.2)
)
print('Num Train Batches: ', len(train_dataloader))
print('Num Valid Batches: ', len(val_dataloader))
print('Num Test Batches: ', len(test_dataloader))