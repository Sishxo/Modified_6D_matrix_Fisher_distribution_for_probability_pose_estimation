import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch_math
from np_math import numdiff
import math

import torch.nn.functional as F

from geometric_utils import torch_batch_quaternion_to_rot_mat, torch_batch_euler_to_rotation
from geometric_utils import numpy_quaternion_to_rot_mat
import torch_norm_factor

epsilon_log_hg = 1e-10

def f_log_hg_torch_approx(S):
    smax = S[:,0]
    v = S[:,1:] / (smax.view(-1, 1) + epsilon_log_hg)
    vsum = torch.sum(v, dim=1)
    sndeg = 0.03125-vsum*0.02083+(0.0104+0.0209*(v[:,0]-v[:,1])**2)*vsum**2
    quot = ((3-vsum)/4)
    d = 2*sndeg / quot
    c = -quot/d
    ret = smax + c - c*torch.exp(-d*smax)
    return ret

class class_log_hg_rough_approx(torch.autograd.Function):
    # same as log_hg_torch_approx for forward, custom backward to get slightly better gradients
    @staticmethod
    def forward(ctx, S):
        smax = S[:,0]
        v = S[:,1:] / (smax.view(-1, 1) + epsilon_log_hg)
        vsum = torch.sum(v, dim=1)
        sndeg = 0.03125-vsum*0.02083+(0.0104+0.0209*(v[:,0]-v[:,1])**2)*vsum**2
        quot = ((3-vsum)/4)
        d = 2*sndeg / quot
        c = -quot/d
        ret = smax + c - c*torch.exp(-d*smax)
        ctx.save_for_backward(S)
        return ret

    @staticmethod
    def backward(ctx, grad):
        S, = ctx.saved_tensors
        tmp = torch.zeros((S.shape[0], 4), device=S.device, dtype=S.dtype)
        tmp[:, :3] = S
        tmp = tmp[:, 0].unsqueeze(1) - tmp
        weight = 1/ (1 + tmp)
        weight = weight / torch.sum(weight, dim=1).unsqueeze(1)
        weight = weight[:, :3]
        return weight * (grad).unsqueeze(1)
log_hg_torch_rough_approx = class_log_hg_rough_approx.apply

def np_log_hg_approx_backward(v):
    tmp = np.zeros(4)
    tmp[:3] = v
    tmp = np.max(tmp) - tmp
    weight = 1 / (1 + tmp)
    weight = weight / np.sum(weight)
    return weight[:3]

def np_log_hg_approx_forward(v_i):
    # note v_i sorted from smallest to largest
    smax = v_i[2]
    v = v_i[:2] / (smax + epsilon_log_hg)
    vsum = np.sum(v)
    sndeg = 0.03125-vsum*0.02083+(0.0104+0.0209*(v[0]-v[1])**2)*vsum**2
    quot = ((3-vsum)/4)
    d = 2*sndeg / quot
    c = -quot/d
    ret = smax + c - c*np.exp(-d*smax)
    return ret

class SinhExprClass(torch.autograd.Function):
    # computes x/sinh(x), custom backward/forward to get numerical stability in x==0
    @staticmethod
    def forward(ctx, input):
        exp_in = torch.exp(input)
        ctx.save_for_backward(input, exp_in)
        ret = 2*input / (exp_in - 1/exp_in)
        ret[torch.abs(input) < 0.01] = 1.0
        return ret

    def backward(ctx, grad_output):
        input, exp_in = ctx.saved_tensors
        exp_in_inv_2 = 1/(exp_in**2)
        ret = 2*(1-input-(1+input)*exp_in_inv_2)/(exp_in*(1-exp_in_inv_2)**2)
        ret[torch.abs(input) < 0.01] = 0
        return ret*grad_output
sinh_expr = SinhExprClass.apply

class LogSinhExprClass(torch.autograd.Function):
    # computes log(sinh(x)/x), custom backward/forward to get numerical stability in x==0 and for large |x|
    @staticmethod
    def forward(ctx, input):
        abs_in = torch.abs(input)
        m_exp_in_2 = torch.exp(-abs_in*2)
        ctx.save_for_backward(input, m_exp_in_2)
        ret = abs_in + torch.log((1-m_exp_in_2)/(abs_in*2))
        ret[abs_in < 0.1] = 0.0
        return ret

    def backward(ctx, grad_output):
        input, m_exp_in_2 = ctx.saved_tensors
        abs_in = torch.abs(input)
        sign_in = torch.sign(input)
        ret = 1 + m_exp_in_2 / (1-m_exp_in_2) - 1/abs_in
        ret[abs_in < 0.1] = 0.0
        return ret*grad_output*sign_in
log_sinh_expr = LogSinhExprClass.apply

def test_sinh_expr():
    x_from = -100
    x_to = 100
    steps = 10000
    steplen = (x_to-x_from)/(steps-1)
    x = np.arange(steps)
    x = x*steplen
    x = x + x_from
    xt = torch.tensor(x, requires_grad=True)
    yt = sinh_expr(xt)
    yt_np = yt.detach().numpy()
    a = torch.sum(yt)
    a.backward()
    y = x/np.sinh(x)
    y[np.abs(x) < 0.001] = 0
    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')

    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')
    plt.show()

    plt.plot(x[:-1] + steplen/2, (y[1:]-y[:-1])/steplen, 'g')
    plt.plot(x, xt.grad.numpy(), 'r--')
    plt.show()

def test_log_sinh_expr():
    x_from = -10
    x_to = 10
    steps = 1000
    steplen = (x_to-x_from)/(steps-1)
    x = np.arange(steps)
    x = x*steplen
    x = x + x_from
    xt = torch.tensor(x, requires_grad=True)
    yt = log_sinh_expr(xt)
    yt_np = yt.detach().numpy()
    a = torch.sum(yt)
    a.backward()
    y = np.log(np.sinh(x)/x)
    y[np.abs(x) < 0.001] = 0
    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')

    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')
    plt.show()

    plt.plot(x[:-1] + steplen/2, (y[1:]-y[:-1])/steplen, 'g')
    plt.plot(x, xt.grad.numpy(), 'r--')
    plt.show()

_global_svd_fail_counter = 0

def quat_R_loss(q, R):
    Rest = torch_batch_quaternion_to_rot_mat(q)
    return torch.sum(((Rest - R)**2).view(-1,9), dim=1), Rest

def direct_quat_loss(q1, q2):
    Rest = torch_batch_quaternion_to_rot_mat(q1)
    q1 = q1 / torch.norm(q1,dim=1).view(-1,1)
    return torch.sum((q1-q2)**2, dim=1), Rest

def euler_loss(angles1, angles2):
    Rest = torch_batch_euler_to_rotation(angles1)
    return torch.sum((angles1-angles2)**2, dim=1), Rest

def angle_error(t_R1, t_R2):
    ret = torch.empty((t_R1.shape[0]), dtype=t_R1.dtype, device=t_R1.device)
    rotation_offset = torch.matmul(t_R1.transpose(1,2), t_R2)
    tr_R = torch.sum(rotation_offset.view(-1,9)[:,::4], axis=1) # batch trace
    cos_angle = (tr_R - 1) / 2
    if torch.any(cos_angle < -1.1) or torch.any(cos_angle > 1.1):
        raise ValueError("angle out of range, input probably not proper rotation matrices")
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle*(180/np.pi)

def angle_error_np(R_1, R_2):
    tr = np.trace(np.matmul(R_1.transpose(), R_2))
    angle = np.arccos((tr-1)/2)*(180.0/np.pi)
    return angle

def numpy_hg_approx1(x,y,z):
    #model with exp as transient
    if z == 0:
        return 0.0
    v0 = x/z
    v1 = y/z
    ss = v0+v1
    sndeg = 0.03125-ss*0.02083+(0.0104+0.0209*(v0-v1)**2)*ss**2
    vs = ss+1
    d = 2*sndeg / (1-vs/4)
    c = -(1-vs/4)/d
    return z + c - c*np.exp(-d*z)

def numpy_hg_approx2(x,y,z):
    #model with c*np.log(1+exp(-d*z)) as transient
    if z == 0:
        return 0.0
    v0 = x/z
    v1 = y/z
    ss = v0+v1
    sndeg = 0.03125-ss*0.02083+(0.0104+0.0209*(v0-v1)**2)*ss**2
    vs = ss+1
    grad0 = 1-vs/4
    d = 4*sndeg/grad0
    c = 2*grad0/d
    return z + c*(np.log(1+np.exp(-d*z))-np.log(2))

class log_sinh_expr_class(torch.autograd.Function):
    # computes log(sinh(x)/x), custom backward/forward to get numerical stability in x==0 and for large |x|
    @staticmethod
    def forward(ctx, input):
        abs_in = torch.abs(input)
        m_exp_in_2 = torch.exp(-abs_in*2)
        ctx.save_for_backward(input, m_exp_in_2)
        ret = abs_in + torch.log((1-m_exp_in_2)/(abs_in*2))
        ret[abs_in < 0.1] = 0.0
        return ret

    def backward(ctx, grad_output):
        input, m_exp_in_2 = ctx.saved_tensors
        abs_in = torch.abs(input)
        sign_in = torch.sign(input)
        ret = 1 + m_exp_in_2 / (1-m_exp_in_2) - 1/abs_in
        ret[abs_in < 0.1] = 0.0
        return ret*grad_output*sign_in

log_sinh_expr = log_sinh_expr_class.apply


def batch_torch_F_to_R(F):
    U,S,V = torch.svd(F)
    with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U,V.transpose(1,2)))
    U[:, :, 2] *= s3sign.view(-1, 1)
    R = torch.matmul(U, V.transpose(1,2))
    return R

def KL_approx_rough(F, R, overreg=1.05):
    global _global_svd_fail_counter
    try:
        U,S,V = torch.svd(F)
        with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            rotation_candidate = torch.matmul(U,V.transpose(1,2))
            s3sign = torch.det(rotation_candidate)
        offset = s3sign*S[:, 2] - S[:, 0] - S[:, 1]
        Diag = torch.empty_like(S)
        Diag[:,0] = 2*(S[:, 0]+S[:, 1])
        Diag[:,1] = 2*(S[:, 0]-s3sign*S[:, 2])
        Diag[:,2] = 2*(S[:, 1]-s3sign*S[:, 2])
        lhg = log_hg_torch_rough_approx(Diag)
        log_norm_factor = lhg + offset
        log_exponent = -torch.matmul(F.view(-1,1,9), R.view(-1, 9,1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
        return log_exponent + overreg*log_norm_factor
    except RuntimeError as e:
        print(e)
        _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100: # we seem to get these problems consistently
            for i in range(F.shape[0]):
                print(F[i])
            raise e
        else:
            return None
        
def KL_approx_sinh(F, R):
    # shared global variable, ugly but these two should not be used at the same time.
    global _global_svd_fail_counter
    # F is bx3x3
    # R is bx3x3
    try:
        _,S,_ = torch.svd(F)
        log_norm_factor = torch.sum(torch.sum(log_sinh_expr(S), dim=1), dim=0)
        log_exponent = -torch.matmul(F.view(-1,1,9), R.view(-1, 9,1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
        return log_exponent + log_norm_factor
    except RuntimeError as e:
        print(e)
        _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100: # we seem to get these problems consistently
            for i in range(F.shape[0]):
                print(F[i])
            raise e
        else:
            return None
        
def KL_Fisher(F, R, overreg=1.05):
    # F is bx3x3
    # R is bx3x3
    global _global_svd_fail_counter
    try:
        U,S,V = torch.svd(F)
        with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            rotation_candidate = torch.matmul(U,V.transpose(1,2))
            s3sign = torch.det(rotation_candidate)
        S_sign = S.clone()
        S_sign[:, 2] *= s3sign
        log_normalizer = torch_norm_factor.logC_F(S_sign)
        log_exponent = -torch.matmul(F.view(-1,1,9), R.view(-1, 9,1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
        return log_exponent + overreg*log_normalizer
    except RuntimeError as e:
        _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100: # we seem to have gotten these problems more often than 10% of batches
            for i in range(F.shape[0]):
                print(F[i])
            raise e
        else:
            print('SVD returned NAN fail counter = {}'.format(_global_svd_fail_counter))
            return None

def total_loss(batch_size, out_F, R_new, R, f_green_R, f_red_R, p_green_R, p_red_R):
    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.0
    
    loss_vmf, Rest = vmf_loss(out_F, R_new, R, overreg=1.025)  
    loss_rot_conf = loss_R_con(batch_size, f_green_R, f_red_R, p_green_R, p_red_R, R)        
    loss_normal = normal_loss(batch_size, f_green_R, f_red_R, p_green_R, p_red_R, R)
    #loss_pm = point_matching_loss(points, f_green_R, f_red_R, p_green_R, p_red_R, R)
    
    losses =  loss_vmf + lambda1 * loss_rot_conf + lambda2 * loss_normal 
    # + lambda3 * loss_pm
    return losses, Rest

# replace the predicted R version
# def vmf_loss(net_F, R_new, R, overreg=1.05):
#     F = net_F.view(-1,3,3)
#     loss_v = loss.KL_Fisher(F, R, overreg=overreg)
#     if loss_v is None:
#         R_new = torch.unsqueeze(torch.eye(3,3, device=R_new.device, dtype=R_new.dtype), 0)
#         R_new = torch.repeat_interleave(R_new, R.shape[0], 0)
#         return None, R_new
#     R_new = loss.batch_torch_F_to_R(F)
#     return loss_v, R_new

def vmf_loss(net_F, R_new, R, overreg=1.05):
    F = net_F.view(-1,3,3)
    loss_v = KL_Fisher(F, R, overreg=overreg)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3,3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, Rest
    Rest = batch_torch_F_to_R(F)
    return loss_v, Rest

def loss_R_con(batch_size, f_green_R, f_red_R, p_green_R, p_red_R, Rs):
    k1 = 13.7
    L1loss = nn.L1Loss(reduction='mean')
    
    gt_green_R, gt_red_R = get_gt_v(batch_size, Rs, axis=2) # Rs=[bs, 3, 3] 
    dis_green = p_green_R - gt_green_R   # bs x 3
    dis_green_norm = torch.norm(dis_green, dim=-1)   # bs
    p_green_con_gt = torch.exp(-k1 * dis_green_norm * dis_green_norm)  # bs
    res_green = L1loss(p_green_con_gt, f_green_R) # p_green_con_gt=[bs], f_green_R=[bs]
    
    dis_red = p_red_R - gt_red_R   # bs x 3
    dis_red_norm = torch.norm(dis_red, dim=-1)   # bs
    p_red_con_gt = torch.exp(-k1 * dis_red_norm * dis_red_norm)  # bs
    res_red = L1loss(p_red_con_gt, f_red_R) # p_red_con_gt=[bs], f_red_R=[bs]
    
    return res_green + res_red

def normal_loss(batch_size, f_green_R, f_red_R, p_green_R, p_red_R, Rs):
    L1loss = nn.L1Loss(reduction='mean')
    gt_green_R, gt_red_R = get_gt_v(batch_size, Rs, axis=2) # gt_green_R, gt_red_R = [bs, 3]

    for i in range(batch_size):
        p_green_R_new, p_red_R_new = get_vertical_rot_vec(f_green_R[i], 
                                                          f_red_R[i], 
                                                          p_green_R[i, ...],  
                                                          p_red_R[i, ...]) 
    p_green_R_new = p_green_R_new.view(1, 3).repeat(batch_size, 1) # p_green_R_new = [bs, 3]
    p_red_R_new = p_red_R_new.view(1, 3).repeat(batch_size, 1) # p_red_R_new = [bs, 3]  
    
    loss_n = L1loss(p_green_R_new, gt_green_R) + L1loss(p_red_R_new, gt_red_R)
    return loss_n

# def point_matching_loss(points, f_green_R, f_red_R, p_green_R, p_red_R, Rs):
    
#     p_green_R_new, p_red_R_new = get_vertical_rot_vec(f_green_R, f_red_R, p_green_R, p_red_R)  
#     Rest = get_rot_mat_y_first(p_red_R_new, p_green_R_new)
    
#     points_est = transform_pts_batch(points, Rest, t=None)
#     points_tgt = transform_pts_batch(points, Rs, t=None)
    
#     weights = 1.0
#     pm_loss = nn.L1Loss(weights * points_est, weights * points_tgt)
#     return pm_loss
    
def get_vertical_rot_vec(c1, c2, y, z):
    ##  c1, c2 are weights
    ##  y, x are rotation vectors
    y = y.view(-1) # bs * dim
    z = z.view(-1)
    
    rot_x = torch.cross(y, z)
    rot_x = rot_x / (torch.norm(rot_x) + 1e-8)
    # cal angle between y and z
    y_z_cos = torch.sum(y * z)
    y_z_theta = torch.acos(y_z_cos)
    theta_2 = c1 / (c1 + c2) * (y_z_theta - math.pi / 2)
    theta_1 = c2 / (c1 + c2) * (y_z_theta - math.pi / 2)
    # first rotate y
    c = torch.cos(theta_1)
    s = torch.sin(theta_1)
    rotmat_y = torch.tensor([[rot_x[0]*rot_x[0]*(1-c)+c, rot_x[0]*rot_x[1]*(1-c)-rot_x[2]*s, rot_x[0]*rot_x[2]*(1-c)+rot_x[1]*s],
                            [rot_x[1]*rot_x[0]*(1-c)+rot_x[2]*s, rot_x[1]*rot_x[1]*(1-c)+c, rot_x[1]*rot_x[2]*(1-c)-rot_x[0]*s],
                            [rot_x[0]*rot_x[2]*(1-c)-rot_x[1]*s, rot_x[2]*rot_x[1]*(1-c)+rot_x[0]*s, rot_x[2]*rot_x[2]*(1-c)+c]]).to(y.device)
    new_y = torch.mm(rotmat_y, y.view(-1, 1))
    # then rotate z
    c = torch.cos(-theta_2)
    s = torch.sin(-theta_2)
    rotmat_z = torch.tensor([[rot_x[0] * rot_x[0] * (1 - c) + c, rot_x[0] * rot_x[1] * (1 - c) - rot_x[2] * s,
                            rot_x[0] * rot_x[2] * (1 - c) + rot_x[1] * s],
                            [rot_x[1] * rot_x[0] * (1 - c) + rot_x[2] * s, rot_x[1] * rot_x[1] * (1 - c) + c,
                            rot_x[1] * rot_x[2] * (1 - c) - rot_x[0] * s],
                            [rot_x[0] * rot_x[2] * (1 - c) - rot_x[1] * s,
                            rot_x[2] * rot_x[1] * (1 - c) + rot_x[0] * s, rot_x[2] * rot_x[2] * (1 - c) + c]]).to(z.device)
    new_z = torch.mm(rotmat_z, z.view(-1, 1))
    return new_y.view(-1), new_z.view(-1)

def get_rot_mat_y_first(y, x):
    # poses
    y = F.normalize(y, p=2, dim=-1)  # bx3
    z = torch.cross(x, y, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    x = torch.cross(y, z, dim=-1)  # bx3
    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)

def get_gt_v(batch_size, Rs, axis=2):
    bs = batch_size  # Rs = bs x 3 x 3

    corners = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.float).to(Rs.device) # corners = [3, 3]
    corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
    
    gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1) # gt_vec = [bs, 9]
    gt_green = gt_vec[:, 3:6] # gt_green = [bs, 3]
    gt_red = gt_vec[:, (6, 7, 8)] # gt_red = [bs, 3]
    
    return gt_green, gt_red