import os, glob
import torch

training_data_dir = "training_data/data"
testing_data_dir = "testing_data/data"
split_dir = "training_data/splits"

def get_split_files(split_name, prefix=""):
    if split_name == 'test':
        files = sorted(glob.glob(os.path.join(prefix + testing_data_dir, '*_datas.pth')))
        return files

    with open(os.path.join(prefix + split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(prefix + training_data_dir, line.strip()) for line in f if line.strip()]
        files = [p + "_datas.pth" for p in prefix]
    return files

def checkpoint_save(model, exp_path, task_name, logger, epoch, save_freq=16, use_cuda=True):
    f = exp_path+'/'+task_name+'-%09d'%epoch + '.pth'
    logger.info('Saving ' + f)
    model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)
    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = exp_path+'/'+task_name+'-%09d'%epoch + '.pth'
    if os.path.isfile(f):
        if not epoch%save_freq==0 and not is_power2(epoch):
            os.remove(f)

def checkpoint_restore(model, exp_path, task_name, logger, use_cuda=True, epoch=0, dist=False, f=''):
    if use_cuda:
        model.cpu()
    if not f:
        if epoch > 0:
            f = exp_path+'/'+task_name+'-%09d'%epoch + '.pth'
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(exp_path+'/'+task_name+'-*.pth'))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path+'/'+task_name+'-') : -4])

    if len(f) > 0:
        if logger != 'logger':
            logger.info('Restore from ' + f)
        checkpoint = torch.load(f)
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

    if use_cuda:
        model.cuda()
    return epoch + 1
