import argparse
import os
import numpy as np
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.visualization import visualize_segmentation
import skimage.io

class Tester(object):
    def __init__(self, args):
        self.args = args
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = args.classes_weights_path
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model = model
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()


        if args.model_path is not None:
            if not os.path.isfile(args.model_path):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.model_path))
            checkpoint = torch.load(args.model_path)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}')".format(args.model_path))



    def test(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            image =  (image.cpu().numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
            prediction = output.data.cpu()
            target = target.cpu().numpy()
            if len(self.val_loader.dataset.class_names) > 2:
                prediction = np.argmax(prediction.numpy(), axis=1)
            else:
                prediction = torch.squeeze(torch.where(prediction > 0.5, torch.ones(prediction.shape), torch.zeros(prediction.shape)),
                                         dim=1).numpy().astype(np.int)
                # Add batch sample into evaluator
            self.evaluator.add_batch(target, prediction)
            if i % 10 == 0:
                for img, pred, gt in zip(image, prediction, target):
                    skimage.io.imsave("./test/deeplab-mobilenet/prob/{}.jpg".format(i), visualize_segmentation(img=img, lbl_true=gt, lbl_pred=pred, n_class=len(self.test_loader.dataset.class_names), label_names=self.test_loader.dataset.class_names))


        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        IoU = self.evaluator.Cls_Intersection_over_Union(1)
        Prec = self.evaluator.Cls_Precision(1)
        Rec = self.evaluator.Cls_Recall(1)
        print('val/total_loss_epoch', test_loss)
        print('val/Acc', Acc)
        print('val/IoU', IoU)
        print('val/Prec', Prec)
        print('val/Rec', Rec)
        print('Validation:')

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # parameters for model
    parser.add_argument('--model', type=str, default='deeplab', choices=['deeplab', 'linknet', 'unet', 'fcn'], help='model name (default: deeplab)')
    parser.add_argument('--backbone', type=str, default='mobilenet', choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--use-balanced-weights', action='store_true', default=False, help='whether to use balanced weights (default: False)')
    parser.add_argument('--classes-weights-path', type=str, default=None, help='path to balanced weights')
    parser.add_argument('--loss-type', type=str, default='bce', choices=['bce', 'ce', 'focal'],
                        help='loss func type (default: bce)')

    # parameters for dataset
    parser.add_argument('--dataset', type=str, default='crack500', choices=['crack500','gapsv1','gapsv2'], help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--train-crop-strategy', type=str, default='rand', choices=['origin','resize', 'rand','prob'], help='train crop strategy')
    parser.add_argument('--val-crop-strategy', type=str, default='origin', choices=['origin','resize','prob'], help='val crop strategy')
    parser.add_argument('--test-crop-strategy', type=str, default='origin', choices=['origin','resize','prob'], help='test crop strategy')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')

    parser.add_argument('--train-batch-size', type=int, default=2, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1, metavar='N', help='input batch size for validation (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: auto)')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # checking point
    parser.add_argument('--model-path', type=str, default='./test/deeplab-mobilenet/prob/model_best.pth.tar', help='put the path to resuming file if needed')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.train_batch_size is None:
        args.train_batch_size = 4 * len(args.gpu_ids)
    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size
    if args.test_batch_size is None:
        args.test_batch_size = args.train_batch_size

    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    tester.test()

if __name__ == "__main__":
   main()
