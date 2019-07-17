import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import make_grid
from cox.utils import Parameters

from . import helpers
from .helpers import AverageMeter, calc_fadein_eps, save_checkpoint, ckpt_at_epoch
from . import constants
import dill
import time

import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

ch = torch

def has_attr(obj, k):
    """Checks both that obj.k exists and is not equal to None"""
    try:
        return (getattr(obj, k) is not None)
    except KeyError as e:
        return False
    except AttributeError as e:
        return False

def sanity_check(args):
    """
    Makes sure all of the required args are present and valid, and fills 
    in defaults for optional args.
    """

    for k in constants.DEFAULTS:
        if not has_attr(args, k) or getattr(args, k) is None:
            print("Using default value for {k}: {dflt}".format(k=k,
                                                               dflt=constants.DEFAULTS[k]))
            setattr(args, k, constants.DEFAULTS[k])

    args.use_best = bool(args.use_best)

    assert has_attr(args, 'train_mode') and (args.train_mode in ['nat', 'adv'])
    if args.train_mode == 'nat':
        required_args = ['epochs', 'log_iters', 'out_dir']
    else:
        required_args = constants.REQUIRED_ARGS

    for k in required_args:
        assert has_attr(args, k), "Cannot find argument {0}".format(k)
    return args

def eval_model(args, loader, model, store):
    args = sanity_check(args)
    writer = store.tensorboard if store else None
    model = torch.nn.DataParallel(model).cuda()
    validate(args, loader, model, 0, True, writer)
    validate(args, loader, model, 0, False, writer)

def train_model(args, model, optimizer, loaders, *, checkpoint=None,
                schedule=None, store=None):
    # write to store tb
    writer = store.tensorboard if store else None

    args = sanity_check(args)
    train_loader, val_loader = loaders
    best_prec1 = 0

    start_epoch = 0
    if checkpoint:
        start_epoch = resume_epoch if resume_epoch is not None else checkpoint['epoch']
        if args.train_mode == 'adv':
            best_prec1 = checkpoint['adv_prec1']
        else:
            best_prec1 = checkpoint['nat_prec1']

    model = torch.nn.DataParallel(model).cuda()
    adv_mode = (args.train_mode == 'adv')

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_prec1, train_loss = train(args, train_loader, model, optimizer,
                                        epoch, adv_mode, writer, store=store)
        last_epoch = (epoch == args.epochs - 1)

        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else
                                          store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_checkpoint_iters
        should_save_ckpt = epoch % save_its == 0 and save_its > 0
        should_log = epoch % args.log_iters == 0
        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            with ch.no_grad():
                prec1, nat_loss = validate(args, val_loader, model, epoch,
                                           False, writer, store=store)

            # loader, model, epoch, input_adv_exs
            adv = args.adv_eval or adv_mode
            adv_val = adv and validate(args, val_loader, model, epoch, True, writer)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = prec1 if (not adv_mode) else adv_prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)

            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss
            }

            if store:
                store[constants.LOGS_TABLE].append_row(log_info)

            if (save_its > 0 and (epoch % save_its == 0)) or last_epoch:
                filename = ckpt_at_epoch(epoch)
                save_checkpoint(filename)

            if last_epoch and store:
                store[constants.CKPTS_TABLE].append_row(sd_info)

            save_checkpoint(constants.CKPT_NAME_LATEST)
            if is_best:
                save_checkpoint(constants.CKPT_NAME_BEST)

        if schedule:
            schedule.step()

        if has_attr(args, 'epoch_hook'):
            args.epoch_hook(model, log_info)

    return model

def train(args, loader, model, optimizer, epoch, adv, writer, store=None):
    return model_loop(args, 'train', loader, model, optimizer, epoch, adv,
                      writer, store=store)

def validate(args, loader, model, epoch, adv, writer, store=None):
    return model_loop(args, 'val', loader, model, None, epoch, adv, writer,
                      store=store)

def model_loop(args, loop_type, loader, model, opt, epoch, adv, writer,
               store=None):
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train mode
    if loop_type == 'train':
        model.train()
        if args.train_mode == 'adv':
            eps = calc_fadein_eps(epoch, args.eps_fadein_iters, args.eps)
            random_restarts = 0
    elif loop_type == 'val':
        model.eval()
        if args.train_mode == 'adv' or adv:
            random_restarts = args.random_restarts
            eps = args.eps

    attacker_criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()

    has_custom_loss = has_attr(args, 'train_criterion')
    if has_custom_loss:
        train_criterion = args.train_criterion
    else:
        train_criterion = ch.nn.CrossEntropyLoss().cuda()

    attack_kwargs = {}
    if args.train_mode == 'adv' or (loop_type == 'val' and adv):
        attack_kwargs = {
            'criterion': attacker_criterion,
            'constraint':args.constraint,
            'eps':eps,
            'step_size':args.attack_lr,
            'iterations':args.attack_steps,
            'random_start':False,
            'random_restarts': random_restarts,
            'use_best': args.use_best
        }
        if has_attr(args, 'custom_loss'):
            attack_kwargs['custom_loss'] = args.custom_loss

        if has_custom_loss:
            attack_kwargs['custom_loss'] = train_criterion

    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
       # measure data loading time
        target = target.cuda(non_blocking=True)
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)

        if len(loss.shape) > 0:
            loss = loss.mean()

        if type(output) is tuple:
            model_logits = output[0]
        else:
            model_logits = output

        # measure accuracy and record loss
        maxk = min(5, model_logits.shape[-1])
        prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))

        losses.update(loss.item(), inp.size(0))
        top1.update(prec1[0], inp.size(0))
        top5.update(prec5[0], inp.size(0))

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term =  args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if loop_type == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()
        elif loop_type == 'val' and adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR
        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1.avg:.3f} | {1}5 {top5.avg:.3f} | '
                'Reg term: {reg} ||'.format( epoch, prec, loop_msg, 
                loss=losses, top1=top1, top5=top5, reg=reg_term))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    return top1.avg, losses.avg

