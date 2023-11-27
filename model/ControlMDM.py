import torch
import torch.nn as nn
from model.mdm import MDM
from data_loaders.humanml.scripts.motion_process import recover_from_ric


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers, norm=None)

    def forward_with_condition(self, src, list_of_controlnet_output, mask=None, src_key_padding_mask=None):
        output = src
        for mod, control_feat in zip(self.layers, list_of_controlnet_output):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            output = output + control_feat
        if self.norm is not None:
            output = self.norm(output)
        return output

    def return_all_layers(self, src, mask=None, src_key_padding_mask=None):
        output = src
        all_layers = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            all_layers.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return all_layers


class ControlMDM(MDM):

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):

        super(ControlMDM, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)
        self.args = args
        self.num_layers = num_layers
        self.multi_person = args.multi_person
        self.upper_orientation_index = [0, 16, 17]  # root, l_shoulder, r_shoulder
        self.lower_orientation_index = [0, 1, 2]  # root, l_hip, r_hip

        # linear layers init with zeros
        if self.dataset == 'kit':
            self.first_zero_linear = nn.Linear(21*3*2 + 2*3, self.latent_dim)
        elif self.dataset == 'humanml':
            self.first_zero_linear = nn.Linear(22*3*2 + 2*3, self.latent_dim)
        else:
            raise NotImplementedError('Supporting only kit and humanml dataset, got {}'.format(self.dataset))
        
        nn.init.zeros_(self.first_zero_linear.weight)
        nn.init.zeros_(self.first_zero_linear.bias)
        self.mid_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)])
        for m in self.mid_zero_linear:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            del self.seqTransEncoder
            self.seqTransEncoder_mdm = TransformerEncoder(seqTransEncoderLayer,
                                                            num_layers=self.num_layers)
            self.seqTransEncoder_control = TransformerEncoder(seqTransEncoderLayer,
                                                             num_layers=self.num_layers)
        else:
            raise ValueError('Supporting only trans_enc arch.')

        self.freeze_block(self.input_process)
        self.freeze_block(self.sequence_pos_encoder)
        self.freeze_block(self.seqTransEncoder_mdm)
        self.freeze_block(self.embed_timestep)
        if 'text' in self.cond_mode:
            self.freeze_block(self.embed_text)
        self.freeze_block(self.output_process)

    def inv_transform(self, data):
        assert self.std is not None and self.mean is not None
        #assert data.requires_grad == True
        std = torch.tensor(self.std, dtype=data.dtype, device=data.device, requires_grad=False)
        mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device, requires_grad=False)
        output = torch.add(torch.mul(data, std), mean)
        return output
    
    def compute_triangle_normals(self, triangles):
        # Compute the vectors from the first point to the other two points
        v1 = triangles[:,:, 1] - triangles[:, :,0]
        v2 = triangles[:,:, 2] - triangles[:,:,0]

        # Compute the cross product of v1 and v2 to get the normal vectors
        normals = torch.cross(v2, v1, dim=-1)

        # Normalize the normal vectors to unit length
        normals = nn.functional.normalize(normals, dim=-1)
        return normals
    
    def humanml_to_global_joint(self, x):
        n_joints = 22 if x.shape[1] == 263 else 21
        curr_joint = self.inv_transform(x.permute(0, 2, 3, 1)).float()
        assert curr_joint.shape[1] == 1
        curr_joint = recover_from_ric(curr_joint, n_joints)
        curr_joint = curr_joint.view(-1, *curr_joint.shape[2:]).permute(0, 2, 3, 1)
        # change root positions for multi-person purpose
        if self.multi_person:
            curr_joint[1::2, :,2,:] *= -1
            curr_joint[1::2, :,0,:] *= -1
            curr_joint[1::2, :,2,:] += 2

            # more than 3 people
            #curr_joint[1, :,2,:] *= -1
            #curr_joint[1, :,0,:] *= -1
            #curr_joint[1, :,2,:] += 2
            #curr_joint[2, :,0,:] += 1
        return curr_joint

    def forward(self, x, timesteps, y=None):
        bs, njoints, nfeats, seqlen = x.shape
        control_bs, n_global_joints, xyz_dim, control_frames = y['global_joint'].shape
        assert bs == control_bs and seqlen == control_frames, "bs {} != {} or seqlen {} != {}".format(bs, control_bs, seqlen, control_frames)
        assert xyz_dim ==3, "xyz_dim {} != 3".format(xyz_dim)
        # prepare global joints for controlmdm
        curr_joint = self.humanml_to_global_joint(x).clone().detach()  # [bs, njoints, 3, seqlen]
        curr_joint.requires_grad = False

        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        # Embed motion to latent space (frame by frame)
        x = self.input_process(x) #[seqlen, bs, d]

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

        # controlmdm
        # orientation
        upper_triangles = curr_joint[:,self.upper_orientation_index,:,:].permute(3,0,1,2)  # [seqlen, bs, 3, 3]
        lower_triangles = curr_joint[:,self.lower_orientation_index,:,:].permute(3,0,1,2)  # [seqlen, bs, 3, 3]
        upper_orientation = self.compute_triangle_normals(upper_triangles)  # [seqlen, bs, 3]
        lower_orientation = self.compute_triangle_normals(lower_triangles)  # [seqlen, bs, 3]

        # relative position to joint
        '''
        relative_position = torch.zeros_like(curr_joint, device = xseq.device, dtype=torch.float32)  # [bs, njoints, 3, seqlen]
        relative_position[1::2,:,:,:] = ((y['global_joint'][::2,:,:,:].unsqueeze(1).float() - \
                                            curr_joint[:,1::2,:,:].unsqueeze(2))*y['global_joint_mask'][::2,:,:,:].bool().float()).float().sum(1)
        relative_position[::2,:,:,:] = ((y['global_joint'][1::2,:,:,:].unsqueeze(1).float() - \
                                            curr_joint[:,::2,:,:].unsqueeze(2))*y['global_joint_mask'][1::2,:,:,:].bool().float()).float().sum(1)
        '''
        relative_position = ((y['global_joint'].float() - curr_joint)*y['global_joint_mask'].bool().float()).float()  # [bs, njoints, 3, seqlen]
        relative_position = relative_position.permute(3, 0, 1, 2).reshape(control_frames, control_bs, -1)  # [seqlen, bs, 22*3]

        # relative position to root
        relative_root = ((y['global_joint'].float() - curr_joint[:,[0],:,:])*y['global_joint_mask'].bool().float()).float()  # [bs, njoints, 3, seqlen]
        relative_root = relative_root.permute(3, 0, 1, 2).reshape(control_frames, control_bs, -1)  # [seqlen, bs, 22*3]
        global_joint_feat = torch.cat((relative_position, relative_root, upper_orientation, lower_orientation), axis=-1)  # [seqlen, bs, 22*3 *2 +3 +3]
        
        global_joint_feat = self.first_zero_linear(global_joint_feat) # [seqlen, bs, d]
        control_input = xseq + torch.cat((torch.zeros_like(emb, device = xseq.device, dtype=torch.float32), global_joint_feat), axis=0)  # [seqlen+1, bs, d]
        control_output_list = self.seqTransEncoder_control.return_all_layers(control_input)  # [seqlen+1, bs, d]
        for i in range(self.num_layers):
            control_output_list[i] = self.mid_zero_linear[i](control_output_list[i])
        
        output = self.seqTransEncoder_mdm.forward_with_condition(xseq, control_output_list)[1:]  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, njoints, nfeats, seqlen]
        return output

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad]
        # return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def trainable_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False

    def unfreeze_block(self, block):
        block.train()
        for p in block.parameters():
            p.requires_grad = True
    
    def forward_without_control(self, x, timesteps, y=None):   #
        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        # Embed motion to latent space (frame by frame)
        x = self.input_process(x) #[seqlen, bs, d]
        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder_mdm(xseq)[1:]  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, njoints, nfeats, seqlen]
        return output