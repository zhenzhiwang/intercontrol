import os
import numpy as np
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

def main():
    out_path = './save/humanml_lr1e-5_all/edit_humanml_lr1e-5_all_000140000_global_joint_seed10_a_person_moves_forward,_then_happily_high-five_another_person_with_his_left_wrist,_and_finally_moves_backward'
    npy_path = os.path.join(out_path, 'results.npy')
    
    npy_dict = np.load(npy_path)
    all_motions = npy_dict['all_motions']
    all_text = npy_dict['all_text']
    all_guidance = npy_dict['all_guidance']
    {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


    for pair_i in range(args.num_samples // 2):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + 2* pair_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            length = max_frames
            motion1 = all_motions[rep_i*args.batch_size + 2* pair_i].transpose(2, 0, 1)[:length]
            motion2 = all_motions[rep_i*args.batch_size + 2* pair_i + 1].transpose(2, 0, 1)[:length]
            guidance = {'mask': all_guidance['mask'][rep_i*args.batch_size + 2*pair_i], 'joint': all_guidance['joint'][rep_i*args.batch_size + 2*pair_i]}
            guidance['joint'] = guidance['joint'].transpose(2, 0, 1)[:length] # seq_len, n_joints, 3
            guidance['mask'] = guidance['mask'].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(2*pair_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({pair_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            plot_3d_motion(animation_save_path, skeleton, motion1, title=caption,
                           dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                           joints2=motion2, painting_features=args.inpainting_mask.split(','), guidance=guidance)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(2*pair_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({2*pair_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()