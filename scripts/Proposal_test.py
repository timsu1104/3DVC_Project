import torch, torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn as ProposalModule
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as init_weight
import argparse, sys, os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time, json

sys.path.append(os.getcwd())
from utils.util import checkpoint_save, checkpoint_restore
from utils.log import init
from utils.eval import evaluate

Total_epochs = 300

def test(model, dataloader):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    with torch.no_grad():
        model = model.eval()
        preds = []
        for batch in tqdm(dataloader):
            rgb = batch['rgb'].cuda()

            input_image = rgb.permute(0, 3, 1, 2)
            images = [image for image in input_image]

            pred = model(images)
            preds += [(box, label, score) for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores'])]

        with open(os.path.join('datasets/2Dproposal.json'), "w") as f:
            json.dump(preds, f)

if __name__ == '__main__':
    exp_path = os.path.join('exp/FPN')

    start = time.time()
    global logger
    logger = init(os.path.join('FPN'))

    # summary writer
    global writer
    writer = SummaryWriter(exp_path)

    from lib.dataloader import Dataset
    model = ProposalModule(weights=init_weight.DEFAULT)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()
    
    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    end_init = time.time()

    ##### dataset
    dataset_ = Dataset()
    dataset_.fpn_testLoader(logger, prepare=True)
    end_data = time.time()
    logger.info("Data Loading time {}".format(end_data - end_init))
    
    ##### resume
    start_epoch = checkpoint_restore(model, exp_path, 'FPN', logger, use_cuda)      # resume from the latest epoch, or specify the epoch to restore
    end_model = time.time()
    
    test(model, dataset_.train_data_loader)
