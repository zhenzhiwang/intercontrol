# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import os
import json
import torch
from diffusion.respace import SpacedDiffusion
from utils.fixseed import fixseed
from utils.parser_util import train_inpainting_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_pretrained_mdm_to_controlmdm
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from data_loaders.humanml_utils import get_control_mask, HML_JOINT_NAMES
from torch.utils.data import DataLoader
from diffusion.control_diffusion import ControlGaussianDiffusion
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from model.ControlMDM import ControlMDM
import numpy as np

def main():
    args = train_inpainting_args()
    assert args.multi_person == False, "multi_person must be False, got"
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                              short_db=args.short_db, cropping_sampler=args.cropping_sampler)
    
    control_joint = args.save_dir.split('/')[-1].split('_')[-2:]
    if control_joint[0]=='left' or control_joint[0]=='right':
        control_joint = control_joint[0] + "_" + control_joint[1]
    else:
        control_joint = control_joint[1]
    assert control_joint in HML_JOINT_NAMES or control_joint == 'all', f"control_joint must be one of {HML_JOINT_NAMES} or 'all', got {control_joint}"
    print(f"control_joint: {control_joint}")
    
    class ControlDataLoader(object):
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            mask_ratio = self.dataset.t2m_dataset.cur_mask_ratio
            for motion, cond in super().__getattribute__('data').__iter__():
                n_joints = 22 if motion.shape[1] == 263 else 21
                 
                unnormalized_motion = self.dataset.t2m_dataset.inv_transform(motion.permute(0, 2, 3, 1)).float()
                global_joints = recover_from_ric(unnormalized_motion, n_joints)
                global_joints = global_joints.view(-1, *global_joints.shape[2:]).permute(0, 2, 3, 1)
                global_joints.requires_grad = False
                cond['y']['global_joint'] = global_joints
                joint_mask = torch.tensor(get_control_mask(args.inpainting_mask, global_joints.shape, joint = control_joint, ratio = mask_ratio, dataset = args.dataset)).to(global_joints.device)
                joint_mask.requires_grad = False
                cond['y']['global_joint_mask'] = joint_mask
                yield motion, cond
        
        def __getattribute__(self, name):
            return super().__getattribute__('data').__getattribute__(name)
        
        def __len__(self):
            return len(super().__getattribute__('data'))
        

    data = ControlDataLoader(data)
    data.dataset.t2m_dataset.cur_mask_ratio = 1.0  # decay mask ratio from 1 to args.mask_ratio if args.mask_ratio < 1
    print("creating model and diffusion...")
    DiffusionClass = ControlGaussianDiffusion if args.filter_noise else SpacedDiffusion
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=ControlMDM, DiffusionClass=DiffusionClass)
    print(f"Loading MDM checkpoints from [{args.pretrained_path}]...")
    state_dict = torch.load(args.pretrained_path, map_location='cpu')
    load_pretrained_mdm_to_controlmdm(model, state_dict)

    diffusion.mean = data.dataset.t2m_dataset.mean
    diffusion.std = data.dataset.t2m_dataset.std
    model.mean = data.dataset.t2m_dataset.mean
    model.std = data.dataset.t2m_dataset.std
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print('Total trainable params: %.2fM' % (sum(p.numel() for p in model.trainable_parameters()) / 1000000.0))
    print("Training...")
    
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
