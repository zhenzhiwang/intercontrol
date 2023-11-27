import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter1d
import torch

def compute_kps_error_with_distance(cur_motion, gt_skel_motions, original_mask):
    '''
    cur_motion [bs, 22, 3, seq_len]
    gt_skel_motions [bs, 22, 3, seq_len]
    mask [bs, 22, 3, seq_len]
    '''
    mask = original_mask.permute(0, 3, 1, 2).bool().float().sum(dim=-1).clamp(0,1)  # [bs, 22, 3, seq_len] -> [bs, seq_len, 22, 3] -> [bs, seq_len, 22]
    if mask.sum()< 1e-6:
        print("mask sum is 0")
        return np.zeros((cur_motion.shape[0], cur_motion.shape[-1])), np.zeros((cur_motion.shape[0], cur_motion.shape[-1]))
    joint_error = (cur_motion.permute(0, 3, 1, 2) - gt_skel_motions.permute(0, 3, 1, 2)) * mask.float()[..., None]
    assert joint_error.shape[-1] == 3
    dist_err = torch.linalg.norm(joint_error, dim=-1) # [bs, seq_len, 22]  joint error norm in xyz dim
    dist_err = dist_err - original_mask.permute(0, 3, 1, 2)[..., 0].float() # [bs, seq_len, 22]
    dist_err = dist_err.clamp(min=0) # [bs, seq_len, 22]
    mask = mask.sum(dim=-1) #[bs, seq_len]
    dist_err = dist_err.sum(dim=-1) # [bs, seq_len]
    dist_err = torch.where(mask > 0, dist_err / mask, torch.tensor(0.0, dtype=torch.float32, device=dist_err.device)) # [bs, seq_len]   joint error average on 22 joints
    mask = mask.clamp(0,1)  # [bs, seq_len]
    return dist_err.detach().cpu().numpy(), mask.detach().cpu().numpy()


def compute_kps_error(cur_motion, gt_skel_motions, original_mask):
    '''
    cur_motion [bs, 22, 3, seq_len]
    gt_skel_motions [bs, 22, 3, seq_len]
    mask [bs, 22, 3, seq_len]
    '''
    mask = original_mask.permute(0, 3, 1, 2).bool().float().sum(dim=-1).clamp(0,1)  # [bs, 22, 3, seq_len] -> [bs, seq_len, 22, 3] -> [bs, seq_len, 22]
    joint_error = (cur_motion.permute(0, 3, 1, 2) - gt_skel_motions.permute(0, 3, 1, 2)) 
    assert joint_error.shape[-1] == 3
    dist_err = torch.linalg.norm(joint_error, dim=-1) * mask.float() # [bs, seq_len, 22]  joint error norm in xyz dim
    mask = mask.sum(dim=-1) #[bs, seq_len]
    dist_err = dist_err.sum(dim=-1) # [bs, seq_len]
    dist_err = torch.where(mask > 0, dist_err / mask, torch.tensor(0.0, dtype=torch.float32, device=dist_err.device)) # [bs, seq_len]   joint error average on 22 joints
    mask = mask.clamp(0,1)  # [bs, seq_len]
    return dist_err.detach().cpu().numpy(), mask.detach().cpu().numpy()

def calculate_trajectory_error(dist_error, mask, strict=True):
    ''' dist_error and mask shape [bs, seq_len]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.

    '''
    assert dist_error.shape == mask.shape
    bs, seqlen = dist_error.shape
    assert np.all(np.logical_or(mask == 0, mask == 1)), "mask array contains values other than 0 or 1"
    mask_sum = mask.sum(-1)  # [bs]
    mean_err_traj = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float)).sum(-1)/ mask_sum, 0.0)

    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - (dist_error * mask.astype(float) <= 0.2).all(1)
        traj_fail_05 = 1.0 - (dist_error * mask.astype(float) <= 0.5).all(1)
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = (mean_err_traj > 0.2).astype(float)
        traj_fail_05 = (mean_err_traj > 0.5).astype(float)
    all_fail_02 = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float) > 0.2).sum(-1) / mask_sum, 0.0)
    all_fail_05 = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float) > 0.5).sum(-1) / mask_sum, 0.0)
    result = np.concatenate([traj_fail_02[..., None], traj_fail_05[..., None], all_fail_02[..., None], all_fail_05[..., None], mean_err_traj[..., None]], axis=1)
     
    assert np.all(~np.isnan(result)), "result array contains NaN values"
    assert result.shape == (bs, 5)
    return result

def calculate_skating_ratio(motions):
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    return skating_ratio, skate_vel


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)