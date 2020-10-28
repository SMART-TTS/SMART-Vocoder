import torch
import os


def actnorm_init(train_loader, model, device):
    x_seed, c_seed = next(iter(train_loader))
    x_seed, c_seed = x_seed.to(device), c_seed.to(device)
    with torch.no_grad():
        model(x_seed, c_seed)

    print('ActNorm is initilized!')

    del x_seed, c_seed


def get_logger(log_path, model_name, test_cll=False, test_speed=False):
    log_eval = open(os.path.join(log_path, '{}.txt'.format('eval')), 'a')

    if test_cll:
        return log_eval
    if test_speed:
        log_speed = open(os.path.join(log_path, '{}.txt'.format('speed')), 'a')
        return log_speed

    log = open(os.path.join(log_path, '{}.txt'.format(model_name)), 'a')
    log_train = open(os.path.join(log_path, '{}.txt'.format('train')), 'a')

    return log, log_train, log_eval


def mkdir(args, synthesize=False, test=False):
    set_desc = f"SMART-Vocoder_fb-{args.n_flow_blocks}_ch-{args.n_channels}_bsz-{args.bsz}"

    if synthesize:
        sample_path = 'synthesize/' + args.model_name + '/' + set_desc +'/temp_' + str(args.temp)
        save_path = 'params/' + args.model_name + '/' + set_desc
        if not os.path.isdir(sample_path):
            os.makedirs(sample_path)
            
        return sample_path, save_path

    if test:
        log_path = 'test_logs/' + args.model_name + '/' + set_desc
        load_path = 'params/' + args.model_name + '/' + set_desc
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        return log_path, load_path

    sample_path = 'samples/' + args.model_name + '/' + set_desc
    save_path = 'params/' + args.model_name + '/' + set_desc
    load_path = 'params/' + args.model_name + '/' + set_desc
    log_path = 'logs/' + args.model_name + '/' + set_desc

    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    return sample_path, save_path, load_path, log_path

def stft(y, scale='linear'):
    D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024)#, window=torch.hann_window(1024).cuda())
    D = torch.sqrt(D.pow(2).sum(-1) + 1e-10)
    # D = torch.sqrt(torch.clamp(D.pow(2).sum(-1), min=1e-10))
    if scale == 'linear':
        return D
    elif scale == 'log':
        S = 2 * torch.log(torch.clamp(D, 1e-10, float("inf")))
        return S
    else:
        pass
