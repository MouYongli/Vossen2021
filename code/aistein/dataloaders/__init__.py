from dataloaders.datasets import crack500
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    """
    :param train_batch_size:
    :param val_batch_size: train/test
    :param test_batch_size: apply cropped image dataset
    """
    if args.dataset == 'crack500':
        train_set = crack500.Crack500Dataset(args, split='train')
        val_set = crack500.Crack500Dataset(args, split='val')
        test_set = crack500.Crack500Dataset(args, split='test')
        num_class = len(train_set.class_names)
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'gaps':
        raise NotImplementedError

    else:
        raise NotImplementedError

