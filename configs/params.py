import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 42
save = True
write_log = True
method = 'CB_Net'
remarks = 'Default'

# Set other parameters
batch_sub = 16
batch_samp = 4
batch_size = batch_sub * batch_samp
epochs = 40
dim = 512

lr = 0.001
lr_sch = [12, 24, 36]
w_decay = 1e-5
dropout = 0.3
momentum = 0.9

# Activate, or deactivate BatchNorm2D
# bn_flag = 0, 1, 2
bn_flag = 1
bn_moment = 0.1
if bn_flag == 1:
    bn_moment = 0.1

# Marginal Cross-Entropy Hyperparameters
cf_s = 128
cf_m = 0.35

# Triplet Loss
tl_id = -1 # -1: use AP, 1: face and peri positive pairs
tl_m = 1.0
tl_k = 1
tl_alpha = 10
tl_ap = 0.001 # 0.01++
if tl_id >= 0:
    tl_ap = 0.0

# Activate / deactivate face_fc, peri_fc w.r.t. face_fc_ce_flag, peri_fc_ce_flag
face_fc_ce_flag = True
peri_fc_ce_flag = True
face_peri_loss_flag = True

if face_fc_ce_flag is True and peri_fc_ce_flag is True and face_peri_loss_flag is False:
    net_descr = 'face_fc w/ CE + peri w/ CE'
    net_tag = str('11_0')
    tl_id = 0
    tl_m = 0.0
    tl_k = 0.0
    tl_alpha = 0.0
elif face_fc_ce_flag is True and peri_fc_ce_flag is True and face_peri_loss_flag is True:
    net_descr = 'face_fc w/ CE + peri w/ CE + Multi-Domain Loss'
    net_tag = str('11_1')
elif face_fc_ce_flag is False and peri_fc_ce_flag is True and face_peri_loss_flag is False:
    net_descr = 'Baseline I : peri_fc w/ CE'
    net_tag = str('01_0B1')
    tl_id = 0
    tl_m = 0.0
    tl_k = 0.0
    tl_alpha = 0.0
elif face_fc_ce_flag is True and peri_fc_ce_flag is False and face_peri_loss_flag is False:
    net_descr = 'Baseline I : face_fc w/ CE'
    net_tag = str('10_0B1')
    tl_id = 0
    tl_m = 0.0
    tl_k = 0.0
    tl_alpha = 0.0
else:
    net_descr = 'unknown'
    raise ValueError("Unknown network parameters.")

bn_moment = float(bn_moment)
dropout = float(dropout)

cf_s = float(cf_s)
cf_m = float(cf_m)
tl_id = int(tl_id)
tl_m = float(tl_m)
tl_k = int(tl_k)
tl_alpha = float(tl_alpha)