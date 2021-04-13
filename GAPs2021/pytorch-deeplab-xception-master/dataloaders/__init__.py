# from datasets import cityscapes, coco, combine_dbs, pascal, sbd, gaps
import sys
sys.path.append('./')
from .datasets import gaps
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'gaps':
        train_set = gaps.GapsDataset(args, split='train')
        val_set = gaps.GapsDataset(args, split='valid-test')
        test_set = gaps.GapsDataset(args, split='test')
        # num_class = len(train_set.class_names)
        if args.merge_labels:
            num_class = 5
        else:
            num_class = 8
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--dataset', type=str, default='gaps',
                        choices=['pascal', 'coco', 'cityscapes, gaps'],
                        help='dataset name (default: gaps)')
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, help='input batch size for testing (default: 16)')
    parser.add_argument('--merge-labels', type=bool, default=True,
                        help='Merge labels (default: True)')
    args = parser.parse_args()
    train, val, test, ncls = make_data_loader(args)
    train_it = iter(train)
    imgs, lbls = next(train_it)
    print(imgs[0, 0, 0:10, 0:10])
    print(ncls)
