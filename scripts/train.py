import torch, torch.optim as optim
import argparse, sys, os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

sys.path.append(os.getcwd())
from utils.util import checkpoint_save, checkpoint_restore
from utils.log import init
from utils.eval import evaluate

Total_epochs = 300
EVAL_FREQ = 5
TAG = ''

def train_epoch(train_loader, model, model_fn, optimizer, exp_path, epoch):
    model.train()
    lossavg = 0
    for batch in train_loader:
        torch.cuda.empty_cache()

        ##### prepare input and forward
        loss, _ = model_fn(batch, model)

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossavg += loss.item()

    lossavg /= len(train_loader)
    logger.info(
        "epoch: {}/{}, train loss: {:.4f}".format(epoch, Total_epochs, lossavg)
    )
    writer.add_scalar('loss_train', lossavg, epoch)
        
    checkpoint_save(model, exp_path, TAG, logger, epoch, 16, use_cuda)
        
def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    with torch.no_grad():
        model.eval()
        lossavg = 0
        gts = []
        preds = []
        for batch in tqdm(val_loader):
            ##### prepare input and forward
            loss, pred = model_fn(batch, model)
            lossavg += loss.item()

            gt = batch["gt"].numpy()
            pred = pred.cpu().numpy()
            gts.append(gt)
            preds.append(pred)

        _, mAcc, _, mIoU = evaluate(gts, preds)
        lossavg /= len(val_loader)
        logger.info("epoch: {}/{}, val loss: {:.4f}, mean accuracy: {:.4f}, mean IoU: {:.4f}".format(epoch, Total_epochs, lossavg/len(val_loader), mAcc, mIoU))
        writer.add_scalar('loss_val', lossavg, epoch)
        writer.add_scalar('Mean acc', mAcc, epoch)
        writer.add_scalar('Mean IoU', mIoU, epoch)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help='', default='')
    parser.add_argument('--label', help='0-78', default='0')
    parser.add_argument('--toy', action='store_true', default=False, help="")
    opt = parser.parse_args()
    TAG = opt.tag
    LBL = opt.label
    TOY = opt.toy
    exp_path = os.path.join('exp/FrustumSegmentationNet', TAG, LBL)

    start = time.time()
    global logger
    logger = init(os.path.join('FrustumSegmentationNet', TAG))

    # summary writer
    global writer
    writer = SummaryWriter(exp_path)

    from model.FrustumSegmentationNet import FrustumSegmentationNet as Network
    from model.FrustumSegmentationNet import model_fn_decorator
    from lib.dataloader import Dataset
    
    model = Network()
    model_fn = model_fn_decorator()

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()
    
    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    end_init = time.time()
    logger.info("Initialization time {}".format(end_init - start))

    ##### dataset
    dataset_ = Dataset()
    dataset_.trainLoader(logger, LBL, TOY)
    dataset_.valLoader(logger, LBL, TOY)
    end_data = time.time()
    
    ##### resume
    start_epoch = checkpoint_restore(model, exp_path, TAG, logger, use_cuda)      # resume from the latest epoch, or specify the epoch to restore
    end_model = time.time()
    
    for epoch in range(start_epoch, Total_epochs+1):
        print("Epoch {}........".format(epoch))
        train_epoch(dataset_.train_data_loader, model, model_fn, optimizer, exp_path, epoch)

        if epoch % EVAL_FREQ == 0:
            eval_epoch(dataset_.val_data_loader, model, model_fn, epoch)
    
    if start_epoch == Total_epochs + 1:
        eval_epoch(dataset_.val_data_loader, model, model_fn, Total_epochs)
