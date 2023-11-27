from diffusion.respace import SpacedDiffusion
from .gaussian_diffusion import _extract_into_tensor, ModelMeanType, ModelVarType
from .control_diffusion import ControlGaussianDiffusion
import torch as th
import numpy as np
import torch.optim as optim
from data_loaders.humanml.scripts.motion_process import recover_from_ric

class MultiControlGaussianDiffusion(ControlGaussianDiffusion):
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        use_posterior=False,
    ):
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """

        assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        global_joints = self.humanml_to_global_joint(x)
        model_kwargs['y']['global_joint'] = th.zeros_like(global_joints, device = x.device)
        model_kwargs['y']['global_joint'][:-1,...] = global_joints[1:,...].clone().detach()
        #model_kwargs['y']['global_joint'][0,...] = global_joints[-1,...].clone().detach()
        model_kwargs['y']['global_joint'].requires_grad = False

        p_mean_variance_func = self.p_mean_variance_bfgs_posterior if use_posterior else self.p_mean_variance_bfgs_x0
        out = p_mean_variance_func(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            k_first = 5,
            k_last = 10,
            t_threshold = 10,
        )
         
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def global_joint_bfgs_optimize(self, x, model_kwargs=None):
        """
        pred_joint: [bs, njoints, 3, seqlen]
        assume interaction between all people in a batch
        """
        assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        pred_joint = self.humanml_to_global_joint(x)

        loss = 0 
        # loss for all time steps
        loss += self.group_motion_region_loss(pred_joint)
        loss += self.avoid_crowd_collision_loss(pred_joint)

        # loss for contact
        contact_mask = model_kwargs['y']['global_joint_mask']
        loss += self.crowd_contact_joint_loss(pred_joint, contact_mask)
        #loss += self.far_away_joint_loss(pred_joint[1::2, :,:,:], pred_joint[::2, :,:,:], far_away_mask)
        return loss
    

    def crowd_contact_joint_loss(self, pred_joint, mask):
        desired_distance = 0.02
        #pred_joint_0 = th.masked_select(pred_joint[0].permute(0,2,1), mask[0, :,:,:].bool().permute(0,2,1)).contiguous().reshape(-1,3)
        #pred_joint_1 = th.masked_select(pred_joint[1].permute(0,2,1), mask[1, :,:,:].bool().permute(0,2,1)).contiguous().reshape(-1,3)
        #pred_joint_2 = th.masked_select(pred_joint[2].permute(0,2,1), mask[2, :,:,:].bool().permute(0,2,1)).contiguous().reshape(-1,3)
        loss = ((pred_joint[0, 11,:,30:50] - pred_joint[1, 11,:,30:50])**2).mean()
        loss += ((pred_joint[1, 11,:,30:50] - pred_joint[2, 11,:,30:50])**2).mean()
        return loss
    
    def humanml_to_global_joint(self, x):
        n_joints = 22 if x.shape[1] == 263 else 21
        pred_joint = self.inv_transform(x.permute(0, 2, 3, 1)).float()
        assert pred_joint.shape[1] == 1
        pred_joint = recover_from_ric(pred_joint, n_joints)
        pred_joint = pred_joint.view(-1, *pred_joint.shape[2:]).permute(0, 2, 3, 1)
        # change root positions for multi-person purpose
        assert pred_joint.shape[0] == 3
        pred_joint[1, :,2,:] *= -1
        pred_joint[1, :,0,:] *= -1
        pred_joint[1, :,2,:] += 2
        pred_joint[2, :,0,:] += 1
        return pred_joint
    
    def group_motion_region_loss(self, pred_joint):
        position = pred_joint[:, 0, [0,2], :]
        cliped_pos = th.clamp(position, -5, 5)
        loss = self.mse_loss(position, cliped_pos) * 0.1
        return loss

    def compute_triangle_normals(self, triangles):
        # Compute the vectors from the first point to the other two points
        v1 = triangles[:, 1] - triangles[:, 0]
        v2 = triangles[:, 2] - triangles[:, 0]

        # Compute the cross product of v1 and v2 to get the normal vectors
        normals = th.cross(v1, v2, dim=1)

        # Normalize the normal vectors to unit length
        normals = th.nn.functional.normalize(normals, dim=1)
        return normals
    
    def avoid_crowd_collision_loss(self, joint):
        root = joint[:, 0, [0,2], :]
        diff = th.norm(root[1:,...] - root[:-1,...], dim = -2)
        loss = th.nn.functional.relu(0.5 - diff).mean()
        diff = th.norm(root[0,...] - root[2,...], dim = -2)
        loss += th.nn.functional.relu(0.5 - diff).mean()
        loss += th.nn.functional.relu(joint[:, 0, 1, :] - 1.1).mean()
        return loss
     
    def get_person_direction(self, joint):
        face_joint_indx = [1, 2, 16, 17]
        l_hip, r_hip, l_shoulder, r_shoulder = face_joint_indx
        across_hip = joint[..., r_hip, :] - joint[..., l_hip, :]
        across_hip = across_hip / across_hip.norm(dim=-1, keepdim=True)
        across_shoulder = joint[..., r_shoulder, :] - joint[..., l_shoulder, :]
        across_shoulder = across_shoulder / across_shoulder.norm(dim=-1, keepdim=True)
        across = (across_hip + across_shoulder) / 2
        across = across / across.norm(dim=-1, keepdim=True)
        y_axis = th.zeros_like(across)
        y_axis[..., 1] = 1
        forward = th.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        return forward

    def face_to_face_loss(self, pred_joint, cond_joint, mask):
        """
        pred_joint: [bs, njoints, 3, seqlen]
        cond_joint: [bs, njoints, 3, seqlen]
        mask: [bs, njoints, 3, seqlen]
        """
        weight={'orientation': 1, 'position': 1, 'hand': 1}
        mask = mask.permute(0, 3, 1, 2).sum(dim=-1).sum(dim=-1).clamp(0,1)
        bs, njoints, ndim, seqlen = pred_joint.shape
        assert ndim == 3, "joint_dim must be 3, got {}".format(ndim)
        pred_joint, cond_joint = pred_joint.permute(0, 3, 1, 2), cond_joint.permute(0, 3, 1, 2)
        direction = self.get_person_direction(pred_joint)
        direction_cond = self.get_person_direction(cond_joint)
        inter_direction = self.get_inter_direction(pred_joint, cond_joint)
        cross_product = (th.cross(direction, inter_direction, dim=-1)[..., 2] + th.cross(inter_direction, direction_cond, dim=-1)[..., 2])/2
        threshold = 0.8
        cross_product[cross_product>threshold] = threshold
        mse_loss = th.nn.MSELoss(reduction='mean')
        position_gt = th.ones_like(cross_product, device = cross_product.device) * threshold
        loss_direction = (direction + direction_cond).abs().mean() * weight['orientation'] 
        loss_position = mse_loss(cross_product, position_gt) * weight['position']
         
        '''
        # hand
        hand_direction = self.get_hand_direction(pred_joint)
        hand_direction_cond = self.get_hand_direction(cond_joint)
        inner_product = (hand_direction[...,:-1] * inter_direction[...,:-1]).sum(dim=-1)
        inner_product_cond =  - (hand_direction_cond[...,:-1] * inter_direction[...,:-1]).sum(dim=-1)
        loss += ((inner_product + inner_product_cond) / 2  * mask).sum() / mask.sum()  * weight['hand']
        '''
        return loss_direction + loss_position