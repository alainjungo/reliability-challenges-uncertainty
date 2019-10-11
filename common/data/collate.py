import torch.utils.data.dataloader as loader


class CollateDict:

    def __init__(self, entries=('labels', 'images')) -> None:
        self.entries = entries

    def __call__(self, batch):
        new_batch = {}
        for key in batch[0]:
            if key in self.entries:
                new_batch[key] = loader.default_collate([d[key] for d in batch])
            else:
                new_batch[key] = [d[key] for d in batch]
        return new_batch
