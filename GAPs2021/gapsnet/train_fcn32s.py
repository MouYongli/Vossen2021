import argparse
import datetime
import os
import os.path as osp
import torch
import yaml

import sys
sys.path.append('../')
from gapsnet.datasets import GapsDataset
from gapsnet.models import FCN32s, VGG16
from gapsnet.trains import Trainer

here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--epochs', type=int, default=200, help='max epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-4, help='learning rate',
    )
    parser.add_argument(
        '--batch-size', type=int, default=2, help='batch size',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    args = parser.parse_args()

    args.model = 'FCN32s'

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(GapsDataset(split='train'), batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(GapsDataset(split='valid-test'), batch_size=args.batch_size, shuffle=False, **kwargs)

    # 2. model
    model = FCN32s(n_class=8)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        epochs=args.epochs,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()