import torch, torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn as ProposalModule
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as init_weight
import argparse, sys, os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

sys.path.append(os.getcwd())
from utils.util import checkpoint_save, checkpoint_restore
from utils.log import init

Total_epochs = 30

def train_epoch(train_loader, model, optimizer, exp_path, epoch):
    model.train()
    lossavg = 0
    for batch in tqdm(train_loader):
        torch.cuda.empty_cache()

        rgb = batch['rgb'].cuda()
        box = batch['box']

        input_image = rgb.permute(0, 3, 1, 2)
        images = [image for image in input_image]
        targets = []
        for i in range(len(images)):
            d = {}
            # print(box.shape)
            # print(box[i, :, :4])
            labels = box[i][:, 4]
            labels[labels >= 79] = -1
            labels += 1
            d['boxes'] = box[i][:, :4].cuda()
            d['labels'] = labels.cuda().long()
            targets.append(d)

        ##### prepare input and forward
        loss = 0
        output = model(images, targets)
        for k, v in output.items():
            loss += v

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
        
    checkpoint_save(model, exp_path, 'FPN', logger, epoch, 16, use_cuda)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', default=False, help="")
    opt = parser.parse_args()
    TOY = opt.toy

    exp_path = os.path.join('exp/FPN_new')

    start = time.time()
    global logger
    logger = init(os.path.join('FPN_new'))

    # summary writer
    global writer
    writer = SummaryWriter(exp_path)

    from lib.dataloader import Dataset
    # model = ProposalModule(weights=init_weight.DEFAULT)
    model = ProposalModule(pretrained=True)
    num_classes = 80
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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
    dataset_.trainLoader(logger, TOY)
    dataset_.valLoader(logger, TOY)
    end_data = time.time()
    logger.info("Data Loading time {}".format(end_data - end_init))
    
    ##### resume
    start_epoch = checkpoint_restore(model, exp_path, 'FPN', logger, use_cuda)      # resume from the latest epoch, or specify the epoch to restore
    end_model = time.time()
    
    for epoch in range(start_epoch, Total_epochs+1):
        print("Epoch {}........".format(epoch))
        train_epoch(dataset_.train_data_loader, model, optimizer, exp_path, epoch)

"""
python scripts/Proposal_train.py
"""