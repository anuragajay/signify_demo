from argparse import ArgumentParser
import traceback

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .helpers import DataPrefetcher
except:
    print(traceback.format_exc())
    raise ValueError("Make sure to run this with python -m (see README.md for details)")

import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR
from . import imagenet_models as models
from torchvision.transforms import ToTensor
ch = torch

import os
import git
from pkg_resources import resource_filename

import cox
import cox.utils
import cox.store

from .datasets import DATASETS
from cox.utils import Parameters

from .train import train_model
from . import constants
from .helpers import ckpt_at_epoch
from .loaders import TransformedLoader

DATASET_TO_CONFIG = {
    'cifar': 'configs/cifar10.json',
    'non_robust_cifar': 'configs/non_robust_cifar10.json',
    'robust_cifar': 'configs/robust_cifar10.json',
    'restricted_imagenet_balanced': 'configs/restricted_imagenet.json',
    'restricted_imagenet': 'configs/restricted_imagenet.json',
    'imagenet': 'configs/imagenet.json',
    'places': 'configs/places.json',
    'places_room': 'configs/places_room.json',
    'places_filter_room': 'configs/places_filter_room.json',
    'sun_lamp': 'configs/sun_lamp.json',
    'cinic': 'configs/cinic10.json',
    'a2b': 'configs/a2b.json'
}

DEFAULT_CONFIG = 'configs/defaults.json'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = ArgumentParser(description='PyTorch ImageNet Training')
# Arguments for train()
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', type=float, help='initial learning rate')
parser.add_argument('--attack-lr', type=str)
parser.add_argument('--attack-steps', type=int)
parser.add_argument('--silent', type=int, choices=[0,1])
parser.add_argument('--eps-fadein-iters', type=int)
parser.add_argument('--random-restarts', type=int)
parser.add_argument('--last-layer-training', type=int, default=0)
parser.add_argument('--adv-eval', type=int, choices=[0,1])
parser.add_argument('--constraint', choices=['inf', '2'])
parser.add_argument('--eps', type=str)
parser.add_argument('--log-iters', type=int, help='how often to log acc etc')
parser.add_argument('--save-checkpoint-iters', type=int, help='if <= 0, dont save at all')

# Arguments for main()
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--arch', '-a')
parser.add_argument('-j', '--workers', type=int, help='number of data loading workers (default:30)')
parser.add_argument('-b', '--batch-size', type=int) 
parser.add_argument('--momentum', type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', type=float, help='weight decay (default:1e-4)')
parser.add_argument('--resume', type=str, help='path to latest checkpoint (default:none)')
parser.add_argument('--resume-epoch', type=int, help='epoch we are resuming from (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=int, choices=[0,1],
                    help='evaluate model on validation set')
parser.add_argument('--train-mode', choices=['nat', 'adv'])
parser.add_argument('--step-lr', type=int, help="If specified, drop interval for a simple step lr")
parser.add_argument('--custom-schedule', help="Custom LR schedule [(milestone, new_lr), ... ]")
parser.add_argument('--dataset', choices=DATASET_TO_CONFIG.keys())
parser.add_argument('--attrs', nargs='+', help="Attribute to classify on (for celebA)")
parser.add_argument('--out-dir')
parser.add_argument('--config-path', help='path to other config file')
parser.add_argument('--exp-name', help='python code for naming experiment')
parser.add_argument('--random-labels', help='whether to use random labels or normal labels', type=float)
parser.add_argument('--use-best', choices=['y', 'n'], 
            help='whether to use best [must be "n" for celebA]')
parser.add_argument('--data-aug', choices=['y', 'n'],
            help='should use data augmentation', default='y')

def model_dataset_from_store(s, overwrite_params={}, which='last'):
    # which options: {'best', 'last', integer}
    if type(s) is tuple:
        s, e = s
        s = cox.store.Store(s, e, mode='r')

    m = s['metadata']
    df = s['metadata'].df

    args = df.to_dict()
    args = {k:v[0] for k,v in args.items()}
    fns = [lambda x: m.get_object(x), lambda x: m.get_pickle(x)]
    conds = [lambda x: m.schema[x] == s.OBJECT, lambda x: m.schema[x] == s.PICKLE]
    for fn, cond in zip(fns, conds):
        args = {k:(fn(v) if cond(k) else v) for k,v in args.items()}

    args.update(overwrite_params)
    args = Parameters(args)

    data_path = os.path.expandvars(args.data)
    if not data_path:
        data_path = '/tmp/'

    dataset = DATASETS[args.dataset](data_path)

    if which == 'last':
        resume = os.path.join(s.path, constants.CKPT_NAME)
    elif which == 'best':
        resume = os.path.join(s.path, constants.CKPT_NAME_BEST)
    else:
        assert isinstance(which, int), "'which' must be one of {'best', 'last', int}"
        resume = os.path.join(s.path, ckpt_at_epoch(which))

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=resume, parallel=False)
    return model, dataset, args


def sanity_check(args):
    if args.dataset == 'celebA':
        assert args.attrs is not None

    if args.evaluate:
        assert args.resume

    # parse fractions horribly if necessary
    if args.eps and (args.train_mode == 'adv'):
        args.eps = eval(args.eps) if args.eps else args.eps
        args.attack_lr = eval(args.attack_lr) if args.attack_lr else None

        if not args.attack_lr:
            args.attack_lr = args.eps / args.attack_steps * 2.5
            tup = (args.attack_lr, args.attack_steps)
            print('no default attack settings:using lr %s with %s steps' % tup)
        elif args.attack_lr and args.attack_steps:
            pass
        else:
            raise ValueError('must specify attack settings!')
    else:
        args.eps = 0

    repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                        search_parent_directories=True)
    git_commit = repo.head.object.hexsha
    args.git_commit = git_commit

    args.use_best = (args.use_best == 'y')

    return args

def main(args, store=None):
    # MAKE DATASET LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path, attrs=args.attrs)

    should_data_aug = args.data_aug == 'y' and (not args.random_labels)
    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=should_data_aug)

    train_loader = DataPrefetcher(train_loader)
    val_loader = DataPrefetcher(val_loader)

    if args.random_labels is not None:
        assert (args.random_labels > 0.) and (args.random_labels <= 1.)
        print("Regenerating dataset with random labels...")
        def transformer(im, targ):
            return im, ch.randint_like(targ, high=dataset.num_classes)
        train_loader = TransformedLoader(train_loader, transformer, ToTensor(),
                                         args.workers, args.batch_size,
                                         do_tqdm=True,
                                         fraction=args.random_labels)

    # MAKE MODEL
    model_kwargs = {
        'arch':args.arch,
        'dataset':dataset,
        'resume_path':args.resume,
        'resume_epoch':args.resume_epoch
    }
    model, checkpoint = make_and_restore_model(**model_kwargs)
    if 'module' in dir(model):
        model = model.module

    # Only Last layer requires gradient if this option is true
    if args.last_layer_training:
        for param in model.parameters():
            param.requires_grad = False
        train_modules = list(model.model.modules())[-2:]
        for module in train_modules:
            for param in module.parameters():
                param.requires_grad = True

    # MAKE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # MAKE SCHEDULE
    # default: fixed learning rate (stuck at initial)
    schedule = LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
    if args.step_lr:
        schedule = StepLR(optimizer, step_size=args.step_lr)
    elif args.custom_schedule:
        cs = args.custom_schedule
        periods = eval(cs) if type(cs) is str else cs
        def lr_func(ep):
            for (milestone, lr) in reversed(periods):
                if ep > milestone: return lr/args.lr
            return args.lr

        schedule = LambdaLR(optimizer, lr_func)

    if args.resume and os.path.isfile(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = args.resume_epoch or 0
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()

    ## TRAIN
    loaders = (train_loader, val_loader)

    print(args)
    model = train_model(args, model, optimizer, loaders, schedule=schedule, store=store)
    return model

if __name__ == "__main__":
    args = None

    args = parser.parse_args()

    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    # use args to fill missing values in JSON
    config_path = resource_filename(__name__, DATASET_TO_CONFIG[args.dataset])
    args = cox.utils.override_json(args, config_path)

    defaults_path = resource_filename(__name__, DEFAULT_CONFIG)
    args = cox.utils.override_json(args, defaults_path)

    # write git commit and set default params if none are filled in
    args = sanity_check(args)

    if args.exp_name:
        exp_name = eval(args.exp_name)
        store = cox.store.Store(args.out_dir, exp_name)
    else:
        store = cox.store.Store(args.out_dir)

    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    logs_schema = constants.LOGS_SCHEMA
    store.add_table(constants.LOGS_TABLE, logs_schema)

    ckpts_schema = constants.CKPTS_SCHEMA
    store.add_table(constants.CKPTS_TABLE, ckpts_schema)

    final_model = main(Parameters(args_dict), store=store)

