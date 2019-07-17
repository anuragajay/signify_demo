import torch as ch
import dill
import os
from . import helpers
from .attacker import AttackerModel

def make_and_restore_model(*_, arch, dataset, resume_path=None, old_format=False, 
        state_dict_path='model', resume_epoch=None, parallel=True):
    """
    make_and_restore_model
    Makes a model and (optionally) restores it from a checkpoint
    - arch (str): Model architecture identifier
    - dataset (Dataset class [see datasets.py])
    - resume_path (str): optional path to checkpoint

    Returns: model (possible loaded with checkpoint), checkpoint
    """
    classifier_model = dataset.get_model(arch)

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = ch.load(resume_path, pickle_module=dill)

            if state_dict_path == 'model' and not ('model' in checkpoint):
                state_dict_path = 'state_dict'
            
            if old_format:
                new_d = dict()
                for i in checkpoint['state_dict'].keys(): 
                    new_d[i.replace('module.','module.model.')] = checkpoint['state_dict'][i]
                    new_d[i.replace('module.','module.attacker.model.')] = checkpoint['state_dict'][i]
            
                new_d['module.normalizer.new_mean'] = dataset.mean[...,None,None]
                new_d['module.normalizer.new_std'] = dataset.std[...,None,None]
                new_d['module.attacker.normalize.new_mean'] = dataset.mean[...,None,None]
                new_d['module.attacker.normalize.new_std'] = dataset.std[...,None,None]
                sd = {k[len('module.'):]:v for k,v in new_d.items()}
            else:
                sd = checkpoint[state_dict_path]
                sd = {k[len('module.'):]:v for k,v in sd.items()}

            model.load_state_dict(sd)
            if parallel:
                model = ch.nn.DataParallel(model)
            model = model.cuda()

            loaded_epoch = resume_epoch if resume_epoch is not None else checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, loaded_epoch))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint