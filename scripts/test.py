import torch
import sys, os, argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.util import checkpoint_restore
from utils.log import init
from lib.dump_helper import dump_result

NUM_OBJECTS = 79

def test(model, model_fn, exp_path, dataloader):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    preds = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            pred = model_fn(batch, model)
            preds.append(pred)
        preds = torch.cat(preds, 0)
    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help='', default='')
    opt = parser.parse_args()
    TAG = opt.tag
    exp_path = os.path.join('exp/FrustumSegmentationNet', TAG)

    global logger
    logger = init(os.path.join('FrustumSegmentationNet', TAG), split='test')

    # summary writer
    global writer
    writer = SummaryWriter(exp_path)

    from model.FrustumSegmentationNet_v2 import FrustumSegmentationNet as Network
    from model.FrustumSegmentationNet_v2 import model_fn_decorator
    from lib.dataloader import Dataset
    
    model = Network()
    model_fn = model_fn_decorator(test=True)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()
    
    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### dataset
    dataset_ = Dataset()
    dataset_.testLoader(logger)

    ##### load model
    checkpoint_restore(model, exp_path, TAG, logger, use_cuda)

    ##### evaluate
    preds = test(model, model_fn, exp_path, dataset_.test_data_loader)
    dump_result(preds)

"""
python scripts/test.py --tag firstversion --label 0
"""