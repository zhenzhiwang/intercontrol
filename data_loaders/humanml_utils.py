import numpy as np
import json
import torch
HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_ROOT_HORIZONTAL_MASK = np.concatenate(([True]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(3)),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

HML_TRAJ_MASK = np.zeros_like(HML_ROOT_MASK)
HML_TRAJ_MASK[1:3] = True

NUM_HML_FEATS = 263

def get_control_mask(mask_name, shape, **kwargs):
    assert mask_name == "global_joint", "mask_name must be 'global_joint', got {}".format(mask_name)
    mask = np.zeros(shape)
    mask = np.maximum(mask, get_global_joint_mask(shape, **kwargs))
    return mask

def select_random_indices(bs, seq_len, num_selected):
    indices = []
    for _ in range(bs):
        indices.append(np.random.choice(seq_len, size=num_selected, replace=False))
    return np.array(indices)

def get_global_joint_mask(shape, joint, ratio=1, dataset = 'humanml'):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_joint (22 for HumanML3D), 3, seq_len))
    """
    bs, num_joint, joint_dim, seq_len = shape
    assert joint_dim == 3, "joint_dim must be 3, got {}".format(joint_dim)
    if dataset == 'humanml':
        assert num_joint == 22, "num_joint must be 22, got {}".format(num_joint)
        if joint == 'all':
            random_joint = np.random.randint(0, num_joint,  size=(1,bs))
        elif joint == 'random_two':
            random_joint = np.random.randint(0, num_joint,  size=(2,bs))
        elif joint == 'random_three':
            random_joint = np.random.randint(0, num_joint,  size=(3,bs))
        else:
            assert joint in HML_JOINT_NAMES, "joint must be one of {}, got {}".format(HML_JOINT_NAMES, joint)
            random_joint = np.ones((1,bs), dtype=int) * HML_JOINT_NAMES.index(joint)
    elif dataset == 'kit':
        assert num_joint == 21, "num_joint must be 21, got {}".format(num_joint)
        if joint == 'all':
            random_joint = np.random.randint(0, num_joint, size=(bs,))
        elif joint == 'pelvis':
            random_joint = np.zeros((bs,), dtype=int)
        else:
            raise NotImplementedError("joint must be one of {} in kit dataset, got {}".format(['all', 'pelvis'], joint))
    else:
        raise NotImplementedError("dataset must be one of {}, got {}".format(['humanml', 'kit'], dataset))
    if np.abs(1 - ratio) < 1e-3:
        random_t = np.ones((bs, 1, 1, seq_len))
    else:
        num_selected = int(ratio * seq_len)
        random_t = np.zeros((bs, 1, 1, seq_len))
        selected_indices = select_random_indices(bs, seq_len, num_selected)
        random_t[np.arange(bs)[:, np.newaxis], :, :, selected_indices] = 1

    random_t = np.tile(random_t, (1, 1, 3, 1))
    mask = np.zeros(shape)
    for i in range(random_joint.shape[0]):
        mask[np.arange(bs)[:, np.newaxis], random_joint[i, :, np.newaxis], :, :] = random_t.astype(float)
    return mask

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_interactive_mask_from_json(list_of_dict, shape):
    model_kwargs = {'y': {'text':[], 'global_joint_mask':[], 'far_away_mask':[]}}
    for i in range(len(list_of_dict)):
        cur_dict = list_of_dict[i]
        model_kwargs['y']['text'].append(cur_dict['text_person1'])
        model_kwargs['y']['text'].append(cur_dict['text_person2'])
        closer_mask_1 = np.zeros(shape)
        closer_mask_2 = np.zeros(shape)
        far_mask_1 = np.zeros(shape)
        far_mask_2 = np.zeros(shape)
        for j in range(len(cur_dict['steps'])):
            cur_joint = cur_dict['steps'][j]
            if cur_joint[4] == 1:
                closer_mask_1[cur_joint[0], :, cur_joint[2]:cur_joint[3]] = cur_joint[5]
                closer_mask_2[cur_joint[1], :, cur_joint[2]:cur_joint[3]] = cur_joint[5]
            else:
                far_mask_1[cur_joint[0],  :, cur_joint[2]:cur_joint[3]] = cur_joint[5]
                far_mask_2[cur_joint[1],  :, cur_joint[2]:cur_joint[3]] = cur_joint[5]
        model_kwargs['y']['global_joint_mask'].append(closer_mask_1)
        model_kwargs['y']['global_joint_mask'].append(closer_mask_2)
        model_kwargs['y']['far_away_mask'].append(far_mask_1)
        model_kwargs['y']['far_away_mask'].append(far_mask_2)
    model_kwargs['y']['global_joint_mask'] = np.array(model_kwargs['y']['global_joint_mask'])
    model_kwargs['y']['far_away_mask'] = np.array(model_kwargs['y']['far_away_mask'])
    bs = 2*len(list_of_dict)
    for i in range(len(list_of_dict)):
        assert np.sum(model_kwargs['y']['global_joint_mask'][2*i, :, :, :].astype(bool)) == np.sum(model_kwargs['y']['global_joint_mask'][2*i+1, :, :, :].astype(bool)), "{}".format(list_of_dict[i]['steps'])
        assert np.sum(model_kwargs['y']['far_away_mask'][2*i, :, :, :].astype(bool)) == np.sum(model_kwargs['y']['far_away_mask'][2*i+1, :, :, :].astype(bool)), "{}".format(list_of_dict[i]['steps'])
    assert bs == model_kwargs['y']['global_joint_mask'].shape[0], "bs must be {}, got {}".format(bs, model_kwargs['y']['global_joint_mask'].shape[0])
    assert bs == model_kwargs['y']['far_away_mask'].shape[0], "bs must be {}, got {}".format(bs, model_kwargs['y']['far_away_mask'].shape[0])
    return model_kwargs, bs


def get_more_people_mask(shape):
    bs = 3
    n_joint, n_xyz, seq_len = shape
    model_kwargs = {'y': {'text':[], 'global_joint_mask':[]}}
    model_kwargs['y']['global_joint_mask']=np.zeros((bs, n_joint, n_xyz, seq_len))
    
    if bs == 3:
        model_kwargs['y']['text'].append("A person steps forward slowly, and hold something with his right wrist and left wrist.")
        model_kwargs['y']['text'].append("A person steps forward slowly, and hold something with his right wrist and left wrist.")
        model_kwargs['y']['global_joint_mask'][0, 21, :,30:50] = 0.05
        model_kwargs['y']['global_joint_mask'][1, 17, :,30:50] = 0.05

        model_kwargs['y']['global_joint_mask'][1, 20, :,60:70] = 0.05
        model_kwargs['y']['global_joint_mask'][2, 16, :,60:70] = 0.05
        
    else:
        pass
    model_kwargs['y']['text'].append("A person steps forward slowly")

    assert bs == model_kwargs['y']['global_joint_mask'].shape[0], "bs must be {}, got {}".format(bs, model_kwargs['y']['global_joint_mask'].shape[0])
    return model_kwargs, bs

    


    
