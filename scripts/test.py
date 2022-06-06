import torch
import sys, os, json, argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.util import checkpoint_save, checkpoint_restore
from utils.log import init

NUM_OBJECTS = 79

def test(model, model_fn, exp_path, dataloader):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    with torch.no_grad():
        model = model.eval()
        preds = {}
        for batch in tqdm(dataloader):
            pred = model_fn(batch, model)

            # Reformat Pred
            scene_name = batch["scene_name"] # (B, )

            for pred_pose, name in zip(pred, scene_name):
                preds[name] = pred_pose.tolist()
        with open(os.path.join(exp_path, 'test_result.json'), "w") as f:
            json.dump(preds, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help='', default='')
    parser.add_argument('--label', help='0-78', default='0')
    opt = parser.parse_args()
    TAG = opt.tag
    LBL = opt.label
    exp_path = os.path.join('exp/PointNet', TAG, LBL)

    global logger
    logger = init(os.path.join('PointNet', TAG), split='test')

    # summary writer
    global writer
    writer = SummaryWriter(exp_path)

    from model.pointnet import PointNet as Network
    from model.pointnet import model_fn_decorator
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
    dataset_.testLoader(logger, LBL)

    ##### load model
    checkpoint_restore(model, exp_path, TAG, logger, use_cuda)

    ##### evaluate
    test(model, model_fn, exp_path, dataset_.test_data_loader)

"""
python scripts/test.py --tag firstversion --label 0
"""