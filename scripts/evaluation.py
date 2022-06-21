import torch, torch.optim as optim
import argparse, sys, os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

sys.path.append(os.getcwd())
from utils.util import checkpoint_restore
from utils.log import init
from utils.eval import evaluate

def eval_epoch(val_loader, model, model_fn, test_model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    with torch.no_grad():
        model.eval()
        lossavg = 0
        gts = []
        preds = []
        for batch in tqdm(val_loader):
            ##### prepare input and forward
            loss, _ = model_fn(batch, model)
            labels = test_model_fn(batch, model)
            lossavg += loss.item()

            gt = batch["gt"].numpy()
            labels = labels.cpu().numpy()
            gts.append(gt)
            preds.append(labels)

        _, mAcc, _, mIoU = evaluate(gts, preds)
        lossavg /= len(val_loader)
        logger.info("epoch: {}/{}, val loss: {:.4f}, mean accuracy: {:.4f}, mean IoU: {:.4f}".format(epoch, EPOCH, lossavg/len(val_loader), mAcc, mIoU))
        writer.add_scalar('loss_val', lossavg, epoch)
        writer.add_scalar('Mean acc', mAcc, epoch)
        writer.add_scalar('Mean IoU', mIoU, epoch)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help='', default='')
    parser.add_argument('--epoch', type=int, default=300, help="")
    opt = parser.parse_args()
    TAG = opt.tag
    EPOCH = opt.epoch
    exp_path = os.path.join('exp/FrustumSegmentationNet', TAG)

    start = time.time()
    global logger
    logger = init(os.path.join('FrustumSegmentationNet', TAG))

    # summary writer
    global writer
    writer = SummaryWriter(exp_path)

    from model.FrustumSegmentationNet_v2 import FrustumSegmentationNet as Network
    from model.FrustumSegmentationNet_v2 import model_fn_decorator
    from lib.dataloader import Dataset
    
    model = Network()
    model_fn = model_fn_decorator()
    test_model_fn = model_fn_decorator(val=True)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()
    
    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    end_init = time.time()
    logger.info("Initialization time {}s".format(end_init - start))

    ##### dataset
    dataset_ = Dataset()
    dataset_.valLoader(logger)
    end_data = time.time()
    logger.info("Data Loading time {}s".format(end_data - end_init))
    
    ##### resume
    start_epoch = checkpoint_restore(model, exp_path, TAG, logger, use_cuda, epoch=EPOCH)      # resume from the latest epoch, or specify the epoch to restore
    end_model = time.time()
    
    eval_epoch(dataset_.val_data_loader, model, model_fn, test_model_fn, EPOCH)

"""
python scripts/train.py --tag test
"""