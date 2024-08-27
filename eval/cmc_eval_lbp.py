import os, glob, sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import torch
import network.mobilefacenet as net
from sklearn.model_selection import KFold
from torch.nn import functional as F
import matplotlib.pyplot as plt
from configs.params import *
import copy
from sklearn.metrics import pairwise
from skimage import feature
import cv2
from sklearn import preprocessing
from tqdm import tqdm
from configs import config
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')
cmc_dict = {}
cmc_avg_dict = {}
cm_cmc_dict_p = {}
cm_cmc_avg_dict_p = {}
cm_cmc_dict_f = {}
cm_cmc_avg_dict_f = {}
mm_cmc_dict = {}
mm_cmc_avg_dict = {}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
dset_name = ['Ethnic', 'Pubfig', 'FaceScrub', 'IMDb Wiki', 'AR']


def create_folder(method):
    lists = ['peri', 'face', 'cm', 'mm']
    boiler_path = './data/cmc/'
    for modal in lists:
        if not os.path.exists(os.path.join(boiler_path, method, modal)):
            os.makedirs(os.path.join(boiler_path, method, modal))

def get_avg(dict_list):
    total_eer = 0
    for items in dict_list:
        total_eer += dict_list[items]
    dict_list['avg'] = total_eer/len(dict_list)

    return dict_list

class LocalBinaryPatterns:
  def __init__(self , numPoints , radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self , image , eps=1e-7):
    # image = cv2.imread(image, 0)
    # cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image , self.numPoints , self.radius)
    # hist = plt.hist(lbp.ravel())
    return lbp # , hist

def preprocess_img(imagePath, type_root):
    type_root = type_root[:4]
    img = cv2.imread(imagePath, 0)    
    desc = LocalBinaryPatterns(8, 2)   

    if type_root == 'face':
        img = cv2.resize(img, (150, 150))
        lbp = desc.describe(img)
        pca = PCA(1) 
    elif type_root == 'peri':
        img = cv2.resize(img, (150, 50))
        lbp = desc.describe(img)
        pca = PCA(3) 
    reduced = pca.fit_transform(lbp)

    return reduced # , hist     

def read_images(type_root, root=None):
    features = torch.Tensor([])
    label_nms = []

    if not root is None:
      dir_root = root
    else:
        dir_root = './data/test1_pearson/' + type_root
    img_path = '**/*.jpg'
    for name in tqdm(glob.glob(os.path.join(dir_root, img_path))):
        feats = preprocess_img(name, type_root)
        feats = torch.unsqueeze(torch.Tensor(feats), dim=0)

        features = torch.cat((features, feats), 0)
        label_nm = name.split('/')[-2]
        label_nms.append(label_nm)
        # print(name)
  
    # convert label text to integers
    le = preprocessing.LabelEncoder()
    labels = torch.Tensor(le.fit_transform(label_nms))

    return torch.flatten(features, start_dim=1), labels

def calculate_cmc(gallery_embedding, probe_embedding, gallery_label, probe_label, last_rank=10):
    """
    :param gallery_embedding: [num of gallery images x embedding size] (n x e) torch float tensor
    :param probe_embedding: [num of probe images x embedding size] (m x e) torch float tensor
    :param gallery_label: [num of gallery images x num of labels] (n x l) torch one hot matrix
    :param label: [num of probe images x num of labels] (m x l) torch one hot matrix
    :param last_rank: the last rank of cmc curve
    :return: (x_range, cmc) where x_range is range of ranks and cmc is probability list with length of last_rank
    """
    gallery_embedding = gallery_embedding.type(torch.float32)
    probe_embedding = probe_embedding.type(torch.float32)
    gallery_label = gallery_label.type(torch.float32)
    probe_label = probe_label.type(torch.float32)


    nof_query = probe_label.shape[0]
    gallery_embedding /= torch.norm(gallery_embedding, p=2, dim=1, keepdim=True)
    probe_embedding /= torch.norm(probe_embedding, p=2, dim=1, keepdim=True)
    prediction_score = torch.matmul(probe_embedding, gallery_embedding.t())
    gt_similarity = torch.matmul(probe_label, gallery_label.t())
    _, sorted_similarity_idx = torch.sort(prediction_score, dim=1, descending=True)
    cmc = torch.zeros(last_rank).type(torch.float32)
    for i in range(nof_query):
        gt_vector = (gt_similarity[i] > 0).type(torch.float32)
        pred_idx = sorted_similarity_idx[i]
        predicted_gt = gt_vector[pred_idx]
        first_gt = torch.nonzero(predicted_gt).type(torch.int)[0]
        if first_gt < last_rank:
            cmc[first_gt:] += 1
    cmc /= nof_query

    if cmc.device.type == 'cuda':
        cmc = cmc.cpu()

    x_range = np.arange(0,last_rank)+1

    return x_range, cmc.numpy()

def calculate_mm_cmc(gallery_embedding_peri, probe_embedding_peri, gallery_label_peri, probe_label_peri, gallery_embedding_face, probe_embedding_face, gallery_label_face, probe_label_face, last_rank=10, mode='concat'):
    gallery_embedding_peri = gallery_embedding_peri.type(torch.float32)
    probe_embedding_peri = probe_embedding_peri.type(torch.float32)
    gallery_label_peri = gallery_label_peri.type(torch.float32)
    probe_label_peri = probe_label_peri.type(torch.float32)
    gallery_embedding_face = gallery_embedding_face.type(torch.float32)
    probe_embedding_face = probe_embedding_face.type(torch.float32)
    gallery_label_face = gallery_label_face.type(torch.float32)
    probe_label_face = probe_label_face.type(torch.float32)

    gallery_label = gallery_label_face
    probe_label = probe_label_face

    if mode == 'concat':
        gallery_embedding = torch.cat((gallery_embedding_face, gallery_embedding_peri), 1)
        probe_embedding = torch.cat((probe_embedding_face, probe_embedding_peri), 1)
        gallery_embedding /= torch.norm(gallery_embedding, p=2, dim=1, keepdim=True)
        probe_embedding /= torch.norm(probe_embedding, p=2, dim=1, keepdim=True)        

    elif mode == 'mean' or mode == 'max':
        gallery_embedding_face = torch.unsqueeze(gallery_embedding_face.detach().cpu(), 0)
        gallery_embedding_peri = torch.unsqueeze(gallery_embedding_peri.detach().cpu(), 0)
        
        gallery_embedding = torch.cat((gallery_embedding_face, gallery_embedding_peri), 0)

        if mode == 'mean':
            gallery_embedding = torch.mean(gallery_embedding, 0)
        elif mode == 'max':
            gallery_embedding = torch.max(gallery_embedding, 0)[0]
        gallery_embedding = F.normalize(gallery_embedding, p=2, dim=1)
        
        probe_embedding_face = torch.unsqueeze(probe_embedding_face.detach().cpu(), 0)
        probe_embedding_peri = torch.unsqueeze(probe_embedding_peri.detach().cpu(), 0)
        
        probe_embedding = torch.cat((probe_embedding_face, probe_embedding_peri), 0)
        
        if mode == 'mean':
            probe_embedding = torch.mean(probe_embedding, 0)
        elif mode == 'max':
            probe_embedding = torch.max(probe_embedding, 0)[0]
        probe_embedding = F.normalize(probe_embedding, p=2, dim=1)

    elif mode == 'score':
        face_dist = pairwise.cosine_similarity(probe_embedding_face, gallery_embedding_face)
        peri_dist = pairwise.cosine_similarity(probe_embedding_peri, gallery_embedding_peri)
  
        prediction_score = torch.Tensor((face_dist + peri_dist) / 2)
        rank_acc = []
        for rank in range(1, last_rank+1):
            probe_pred, pred_indices = torch.topk(prediction_score, rank, dim=1)
            gallery_label_ = torch.where(gallery_label)[1]
            probe_pred = gallery_label_[pred_indices]
            probe_label_ = torch.where(probe_label)[1]
            probe_acc = torch.sum(torch.sum(probe_pred.eq(probe_label_.view(-1,1)), dim=1, keepdims=True) >=1 )/ probe_label_.shape[0]
            rank_acc.append(probe_acc)

    nof_query = probe_label_peri.shape[0]
    
    if not mode == 'score':
        prediction_score = torch.matmul(probe_embedding, gallery_embedding.t())
    gt_similarity = torch.matmul(probe_label, gallery_label.t())
    _, sorted_similarity_idx = torch.sort(prediction_score, dim=1, descending=True)
    cmc = torch.zeros(last_rank).type(torch.float32)
    for i in range(nof_query):
        gt_vector = (gt_similarity[i] > 0).type(torch.float32)
        pred_idx = sorted_similarity_idx[i]
        predicted_gt = gt_vector[pred_idx]
        first_gt = torch.nonzero(predicted_gt).type(torch.int)[0]
        if first_gt < last_rank:
            cmc[first_gt:] += 1
    cmc /= nof_query
    if cmc.device.type == 'cuda':
        cmc = cmc.cpu()

    x_range = np.arange(0, last_rank)+1

    return x_range, cmc.numpy()

def cmc_extractor(root_pth=config.evaluation['identification'], modal='periocular', peri_flag=True, rank=10):
    total_cmc = np.empty((0, rank), int) 
    modal = modal[:4]
    for datasets in dset_list:
        cmc_lst = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'        
        modal_root = '/' + modal[:4] + '/'    
        data_loader_dict = {} 
        data_lbl_dict = {}   

        # data loader and datasets
        if not datasets in ['ethnic',]:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('/')[-1]
                gallery_probe = base_nm.split('/')[-1]
                modal_base = directs + modal_root
                feats, lbls = read_images(type_root=modal, root=modal_base)
                data_loader_dict[gallery_probe] = feats
                data_lbl_dict[gallery_probe] = lbls
            # print(datasets)       

            # split data loaders into folds
            kf = KFold(n_splits=len(data_loader_dict))
            for probes, gallery in kf.split(data_loader_dict):
                for i in range(len(probes)):
                    peri_fea_gal, peri_lbl_gal = data_loader_dict[list(data_loader_dict)[int(gallery)]], data_lbl_dict[list(data_lbl_dict)[int(gallery)]]
                    peri_fea_pr, peri_lbl_pr = data_loader_dict[list(data_loader_dict)[probes[i]]], data_lbl_dict[list(data_lbl_dict)[probes[i]]]
                    # peri_fea_gal, peri_fea_pr = torch.flatten(peri_fea_gal, start_dim=1), torch.flatten(peri_fea_pr, start_dim=1)
                    peri_lbl_pr, peri_lbl_gal = F.one_hot(peri_lbl_pr.unsqueeze(0).to(torch.int64)), F.one_hot(peri_lbl_gal.unsqueeze(0).to(torch.int64))
                    peri_lbl_pr, peri_lbl_gal = torch.squeeze(peri_lbl_pr, 0), torch.squeeze(peri_lbl_gal, 0)
                    rng, cmc = calculate_cmc(peri_fea_gal, peri_fea_pr, peri_lbl_gal, peri_lbl_pr, last_rank=rank)
                    cmc_lst = np.append(cmc_lst, np.array([cmc]), axis=0)
                
            cmc = np.mean(cmc_lst, axis=0)
        
        # *** ***
        elif datasets == 'ethnic':
            ethnic_fea_gal, ethnic_lbl_gal = read_images(type_root=modal, root=(root_pth + 'ethnic/Recognition/gallery/' + modal[:4] + '/'))
            ethnic_fea_pr, ethnic_lbl_pr = read_images(type_root=modal, root=(root_pth + 'ethnic/Recognition/probe/' + modal[:4] + '/'))   

            # ethnic_fea_gal, ethnic_fea_pr = torch.flatten(ethnic_fea_gal, start_dim=1), torch.flatten(ethnic_fea_pr, start_dim=1)
            ethnic_lbl_pr, ethnic_lbl_gal = F.one_hot(ethnic_lbl_pr.unsqueeze(0).to(torch.int64)), F.one_hot(ethnic_lbl_gal.unsqueeze(0).to(torch.int64))
            ethnic_lbl_pr, ethnic_lbl_gal = torch.squeeze(ethnic_lbl_pr, 0), torch.squeeze(ethnic_lbl_gal, 0)  

            rng, cmc = calculate_cmc(ethnic_fea_gal, ethnic_fea_pr, ethnic_lbl_gal, ethnic_lbl_pr, last_rank=rank)

        cmc_dict[datasets] = cmc
        print(datasets, cmc)
        # print(cmc_dict)
    for ds in cmc_dict:
        total_cmc = np.append(total_cmc, np.array([cmc_dict[ds]]), axis=0)
    cmc_avg_dict['avg'] = np.mean(total_cmc, axis=0)

    return cmc_dict, cmc_avg_dict, rng

def cm_cmc_extractor(root_pth=config.evaluation['identification'], rank=10):
    total_cmc_f = np.empty((0, rank), int) 
    total_cmc_p = np.empty((0, rank), int) 

    for datasets in dset_list:
        cmc_lst_f = np.empty((0, rank), int)
        cmc_lst_p = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'     
        peri_data_loaders = []
        peri_data_sets = []     
        face_data_loaders = []
        face_data_sets = []            
        face_dl_dict = {}
        face_lbl_dict = {}
        peri_dl_dict = {}
        peri_lbl_dict = {}

        # data loader and datasets
        if not datasets in ['ethnic']:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('/')[-1]
                if base_nm == 'gallery':
                    peri_fea_gal, peri_lbl_gal = read_images(type_root='peri', root=(directs + '/peri/'))
                    face_fea_gal, face_lbl_gal = read_images(type_root='face', root=(directs + '/face/'))
                    face_lbl_gal = F.one_hot(face_lbl_gal.to(torch.int64))
                    peri_lbl_gal = F.one_hot(peri_lbl_gal.to(torch.int64))

                else:
                    peri_fea_pr, peri_lbl_pr = read_images(type_root='peri', root=(directs + '/peri/'))
                    face_fea_pr, face_lbl_pr = read_images(type_root='face', root=(directs + '/face/'))
                    peri_dl_dict[base_nm] = peri_fea_pr
                    face_dl_dict[base_nm] = face_fea_pr
                    peri_lbl_dict[base_nm] = peri_lbl_pr
                    face_lbl_dict[base_nm] = face_lbl_pr

        # print(datasets)        
        # # *** ***               
            for probes in peri_dl_dict:     
                peri_lbl_pr  = F.one_hot(peri_lbl_dict[probes].to(torch.int64))               
                rng_f, cmc_f = calculate_cmc(face_fea_gal, peri_dl_dict[probes], face_lbl_gal, peri_lbl_pr, last_rank=rank)
                cmc_lst_f = np.append(cmc_lst_f, np.array([cmc_f]), axis=0)

            for probes in face_dl_dict:      
                face_lbl_pr  = F.one_hot(face_lbl_dict[probes].to(torch.int64))               
                rng_p, cmc_p = calculate_cmc(peri_fea_gal, face_dl_dict[probes], peri_lbl_gal, face_lbl_pr, last_rank=rank)
                cmc_lst_p = np.append(cmc_lst_p, np.array([cmc_p]), axis=0)
                
            cmc_f = np.mean(cmc_lst_f, axis=0)
            cmc_p = np.mean(cmc_lst_p, axis=0)

        elif datasets == 'ethnic':
            p_ethnic_fea_gal, p_ethnic_lbl_gal = read_images(type_root='peri', root=(root_pth + 'ethnic/Recognition/gallery/peri/'))
            p_ethnic_fea_pr, p_ethnic_lbl_pr = read_images(type_root='peri', root=(root_pth + 'ethnic/Recognition/probe/peri/'))
            p_ethnic_lbl_pr, p_ethnic_lbl_gal = F.one_hot(p_ethnic_lbl_pr.unsqueeze(0).to(torch.int64)), F.one_hot(p_ethnic_lbl_gal.unsqueeze(0).to(torch.int64))
            p_ethnic_lbl_pr, p_ethnic_lbl_gal = torch.squeeze(p_ethnic_lbl_pr, 0), torch.squeeze(p_ethnic_lbl_gal, 0)

            f_ethnic_fea_gal, f_ethnic_lbl_gal = read_images(type_root='face', root=(root_pth + 'ethnic/Recognition/gallery/face/'))
            f_ethnic_fea_pr, f_ethnic_lbl_pr = read_images(type_root='face', root=(root_pth + 'ethnic/Recognition/probe/face/'))
            f_ethnic_lbl_pr, f_ethnic_lbl_gal = F.one_hot(f_ethnic_lbl_pr.unsqueeze(0).to(torch.int64)), F.one_hot(f_ethnic_lbl_gal.unsqueeze(0).to(torch.int64))
            f_ethnic_lbl_pr, f_ethnic_lbl_gal = torch.squeeze(f_ethnic_lbl_pr, 0), torch.squeeze(f_ethnic_lbl_gal, 0)

            rng_f, cmc_f = calculate_cmc(f_ethnic_fea_gal, p_ethnic_fea_pr, f_ethnic_lbl_gal, p_ethnic_lbl_pr, last_rank=rank)
            rng_p, cmc_p = calculate_cmc(p_ethnic_fea_gal, f_ethnic_fea_pr, p_ethnic_lbl_gal, f_ethnic_lbl_pr, last_rank=rank)

        cm_cmc_dict_f[datasets] = cmc_f
        print(datasets, cmc_f)
        cm_cmc_dict_p[datasets] = cmc_p
        print(datasets, cmc_p)

    for ds in cm_cmc_dict_f:
        total_cmc_f = np.append(total_cmc_f, np.array([cm_cmc_dict_f[ds]]), axis=0)
    cm_cmc_avg_dict_f['avg'] = np.mean(total_cmc_f, axis=0)

    for ds in cm_cmc_dict_p:
        total_cmc_p = np.append(total_cmc_p, np.array([cm_cmc_dict_p[ds]]), axis=0)
    cm_cmc_avg_dict_p['avg'] = np.mean(total_cmc_p, axis=0)

    return cm_cmc_dict_f, cm_cmc_avg_dict_f, cm_cmc_dict_p, cm_cmc_avg_dict_p

def mm_cmc_extractor(root_pth=config.evaluation['identification'], rank=10, mode='concat'):
    total_cmc = np.empty((0, rank), int) 
    for datasets in dset_list:
        cmc_lst = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'     
        face_data_loader_dict = {} 
        face_data_lbl_dict = {}   
        peri_data_loader_dict = {} 
        peri_data_lbl_dict = {}   

        # data loader and datasets
        if not datasets in ['ethnic']:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('/')[-1]
                gallery_probe = base_nm.split('/')[-1]
                peri_feats, peri_lbls = read_images(type_root='periocular', root=directs + '/peri/')
                face_feats, face_lbls = read_images(type_root='face', root=directs + '/face/')
                face_data_loader_dict[gallery_probe] = face_feats
                face_data_lbl_dict[gallery_probe] = face_lbls
                peri_data_loader_dict[gallery_probe] = peri_feats
                peri_data_lbl_dict[gallery_probe] = peri_lbls
        # print(datasets)        
        # *** ***
            # split data loaders into folds
            kf = KFold(n_splits=len(peri_data_loader_dict))
            for probes, gallery in kf.split(peri_data_loader_dict):                
                for i in range(len(probes)):
                    peri_fea_gal, peri_lbl_gal = peri_data_loader_dict[list(peri_data_loader_dict)[int(gallery)]], peri_data_lbl_dict[list(peri_data_lbl_dict)[int(gallery)]]
                    peri_fea_pr, peri_lbl_pr = peri_data_loader_dict[list(peri_data_loader_dict)[probes[i]]], peri_data_lbl_dict[list(peri_data_lbl_dict)[probes[i]]]
                    peri_lbl_pr, peri_lbl_gal = F.one_hot(peri_lbl_pr.unsqueeze(0).to(torch.int64)), F.one_hot(peri_lbl_gal.unsqueeze(0).to(torch.int64))
                    peri_lbl_pr, peri_lbl_gal = torch.squeeze(peri_lbl_pr, 0), torch.squeeze(peri_lbl_gal, 0)

                    face_fea_gal, face_lbl_gal = face_data_loader_dict[list(face_data_loader_dict)[int(gallery)]], face_data_lbl_dict[list(face_data_lbl_dict)[int(gallery)]]
                    face_fea_pr, face_lbl_pr = face_data_loader_dict[list(face_data_loader_dict)[probes[i]]], face_data_lbl_dict[list(face_data_lbl_dict)[probes[i]]]
                    face_lbl_pr, face_lbl_gal = F.one_hot(face_lbl_pr.unsqueeze(0).to(torch.int64)), F.one_hot(face_lbl_gal.unsqueeze(0).to(torch.int64))
                    face_lbl_pr, face_lbl_gal = torch.squeeze(face_lbl_pr, 0), torch.squeeze(face_lbl_gal, 0)

                    rng, cmc = calculate_mm_cmc(peri_fea_gal, peri_fea_pr, peri_lbl_gal, peri_lbl_pr, face_fea_gal, face_fea_pr, face_lbl_gal, face_lbl_pr, last_rank=rank, mode=mode)
                    cmc_lst = np.append(cmc_lst, np.array([cmc]), axis=0)
                
            cmc = np.mean(cmc_lst, axis=0)
        
        elif datasets == 'ethnic':
            f_ethnic_fea_gal, f_ethnic_lbl_gal = read_images(type_root='face', root=(root_pth + 'ethnic/Recognition/gallery/face/'))
            f_ethnic_fea_pr, f_ethnic_lbl_pr = read_images(type_root='face', root=(root_pth + 'ethnic/Recognition/probe/face/'))
            p_ethnic_fea_gal, p_ethnic_lbl_gal = read_images(type_root='periocular', root=(root_pth + 'ethnic/Recognition/gallery/peri/'))
            p_ethnic_fea_pr, p_ethnic_lbl_pr = read_images(type_root='periocular', root=(root_pth + 'ethnic/Recognition/probe/peri/'))

            f_ethnic_lbl_gal, f_ethnic_lbl_pr = F.one_hot(f_ethnic_lbl_gal.unsqueeze(0).to(torch.int64)), F.one_hot(f_ethnic_lbl_pr.unsqueeze(0).to(torch.int64))
            f_ethnic_lbl_gal, f_ethnic_lbl_pr = torch.squeeze(f_ethnic_lbl_gal, 0), torch.squeeze(f_ethnic_lbl_pr, 0)  
            p_ethnic_lbl_gal, p_ethnic_lbl_pr = F.one_hot(p_ethnic_lbl_gal.unsqueeze(0).to(torch.int64)), F.one_hot(p_ethnic_lbl_pr.unsqueeze(0).to(torch.int64))
            p_ethnic_lbl_gal, p_ethnic_lbl_pr = torch.squeeze(p_ethnic_lbl_gal, 0), torch.squeeze(p_ethnic_lbl_pr, 0)  
        
            rng, cmc = calculate_mm_cmc(p_ethnic_fea_gal, p_ethnic_fea_pr, p_ethnic_lbl_gal, p_ethnic_lbl_pr, f_ethnic_fea_gal, f_ethnic_fea_pr, f_ethnic_lbl_gal, f_ethnic_lbl_pr, 
                                            last_rank=rank, mode = mode)

        mm_cmc_dict[datasets] = cmc
        print(datasets, cmc)
        # print(cmc_dict)
    for ds in cmc_dict:
        total_cmc = np.append(total_cmc, np.array([cmc_dict[ds]]), axis=0)
    # cmc_dict['avg'] = np.mean(total_cmc, axis=0)
    mm_cmc_avg_dict['avg'] = np.mean(total_cmc, axis=0)

    return mm_cmc_dict, mm_cmc_avg_dict, rng

if __name__ == '__main__':
    method = 'lbp'
    rank = 10
    rng = np.arange(0, rank)+1
    mm_mode = 'concat'
    create_folder(method)

    # Compute CMC values (Periocular)
    peri_cmc_dict, peri_avg_dict, peri_rng = cmc_extractor(root_pth=config.evaluation['identification'], modal='periocular', peri_flag=True, rank=rank)
    peri_cmc_dict = copy.deepcopy(peri_cmc_dict)
    peri_avg_dict = copy.deepcopy(peri_avg_dict)
    print('LBP (Periocular): \n', peri_cmc_dict)
    torch.save(peri_cmc_dict, './data/cmc/' + str(method) + '/peri/peri_cmc_dict.pt')        
    torch.save(peri_avg_dict, './data/cmc/' + str(method) + '/peri/peri_avg_dict.pt')

    # Compute CMC values (Face)
    face_cmc_dict, face_avg_dict, face_rng = cmc_extractor(root_pth=config.evaluation['identification'], modal='face', peri_flag=False, rank=rank)
    face_cmc_dict = copy.deepcopy(face_cmc_dict)
    face_avg_dict = copy.deepcopy(face_avg_dict)
    print('LBP (Face): \n', face_cmc_dict)    
    torch.save(face_cmc_dict, './data/cmc/' + str(method) + '/face/face_cmc_dict.pt')
    torch.save(face_avg_dict, './data/cmc/' + str(method) + '/face/face_avg_dict.pt')        

    # Compute CMC Values (Cross-Modal)
    cm_cmc_dict_f, cm_avg_dict_f, cm_cmc_dict_p, cm_avg_dict_p = cm_cmc_extractor(root_pth=config.evaluation['identification'], rank=rank)
    cm_cmc_dict_f = copy.deepcopy(cm_cmc_dict_f)
    cm_avg_dict_f = copy.deepcopy(cm_avg_dict_f)
    cm_cmc_dict_p = copy.deepcopy(cm_cmc_dict_p)
    cm_avg_dict_p = copy.deepcopy(cm_avg_dict_p)  
    print('LBP (Cross-Modal): \n', cm_cmc_dict_f, cm_cmc_dict_p)
    torch.save(cm_cmc_dict_f, './data/cmc/' + str(method) + '/cm/cm_base_cmc_dict_f.pt')
    torch.save(cm_avg_dict_f, './data/cmc/' + str(method) + '/cm/cm_base_avg_dict_f.pt')
    torch.save(cm_cmc_dict_p, './data/cmc/' + str(method) + '/cm/cm_base_cmc_dict_p.pt')
    torch.save(cm_avg_dict_p, './data/cmc/' + str(method) + '/cm/cm_base_avg_dict_p.pt')