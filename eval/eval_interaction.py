from diffusion.two_person_control_diffusion import TwoPeopleControlGaussianDiffusion
from utils.parser_util import evaluation_inpainting_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_llm_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import load_controlmdm_and_diffusion
from model.ControlMDM import ControlMDM
from data_loaders.humanml_utils import get_interactive_mask_from_json
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import wrap_model

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_interaction(motion_loaders, file):
    trajectory_score_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    print('========== Evaluating Interaction Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_size = 0
        skate_ratio_sum = 0.0
        traj_err = []
        traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "loc_fail_20cm", "loc_fail_50cm", "avg_err(m)"]
        # print(motion_loader_name)
        assert motion_loader_name == 'vald'  # tested method named vald as default
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                skate_ratio, err_np = batch
                all_size += skate_ratio.shape[0]
                traj_err.append(err_np)
                skate_ratio_sum += skate_ratio.sum()

        ### For trajecotry evaluation ###
        traj_err = np.concatenate(traj_err).mean(0)
        trajectory_score_dict[motion_loader_name] = traj_err
        line = f'---> [{motion_loader_name}] Traj Error: '
        print(line)
        print(line, file=file, flush=True)
        line = ''
        for (k, v) in zip(traj_err_key, traj_err):
            line += '    (%s): %.4f \n' % (k, np.mean(v))
        print(line)
        print(line, file=file, flush=True)

        # For skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score
        print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
        print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)
    return trajectory_score_dict, skating_ratio_dict

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def evaluation(eval_motion_loaders, log_file, replication_times):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Trajectory Error': OrderedDict({}),
                                    'Skating Ratio': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader= motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            trajectory_score_dict, skating_ratio_dict = evaluate_interaction(motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in trajectory_score_dict.items():
                if key not in all_metrics['Trajectory Error']:
                    all_metrics['Trajectory Error'][key] = [item]
                else:
                    all_metrics['Trajectory Error'][key] += [item]

            for key, item in skating_ratio_dict.items():
                if key not in all_metrics['Skating Ratio']:
                    all_metrics['Skating Ratio'][key] = [item]
                else:
                    all_metrics['Skating Ratio'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif metric_name == 'Trajectory Error':
                    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "loc_fail_20cm", "loc_fail_50cm", "avg_err(m)"]
                    line = f'---> [{model_name}]'
                    print(line)
                    print(line, file=f, flush=True)
                    line = ''
                    for i in range(len(mean)): # zip(traj_err_key, mean):
                        line += '    (%s): Mean: %.4f CInt: %.4f; \n' % (traj_err_key[i], mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == '__main__':
    args = evaluation_inpainting_parser()
    assert args.multi_person == True, 'only for multi-person !'
    assert args.guidance_param == 2.5
    assert args.inpainting_mask == 'global_joint', "This script only supports global_joint inpainting!"
    fixseed(args.seed)
    args.batch_size = 32
    model_name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dataset_name = args.dataset
    log_file = os.path.join(os.path.dirname(args.model_path), 'interactive_niter_' + str(int(niter)) +'_'+ args.control_joint)
    log_file += f'_mask{args.mask_ratio}'
    log_file += f'_bfgs_first{args.bfgs_times_first}_last{args.bfgs_times_last}_skip{args.bfgs_interval}'
    if args.use_posterior:
        log_file += '_posterior'
    else:
        log_file += '_x0'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')
    assert args.overwrite or not os.path.exists(log_file), "Log file already exists!"

    print(f'Eval mode [{args.eval_mode}]')
    replication_times = 10 if args.replication_times is None else args.replication_times # about 12 Hrs

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating humanml loader...")
    split = 'test'
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='eval', size=1)  # humanml only for loading diffusion model config

    logger.log("Creating model and diffusion...")
    DiffusionClass =  TwoPeopleControlGaussianDiffusion
    model, diffusion = load_controlmdm_and_diffusion(args, gen_loader, dist_util.dev(), ModelClass=ControlMDM, DiffusionClass=DiffusionClass)
    diffusion.mean = gen_loader.dataset.t2m_dataset.mean
    diffusion.std = gen_loader.dataset.t2m_dataset.std
    max_motion_length = 100
    eval_motion_loaders = {
        ################
        ## LLM generated joint-joint pair dataset##
        ################
        'vald': lambda: get_llm_loader(
            args, model, diffusion, args.batch_size, max_motion_length, 
            args.guidance_param, args.interaction_json,
        )
    }
    evaluation(eval_motion_loaders, log_file, replication_times)