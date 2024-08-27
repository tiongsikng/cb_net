import os, sys
sys.path.insert(0, os.path.abspath('.'))

import torch
import os, glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise
from data import data_loader
from network import load_model
import network.mobilefacenet as net
from training import train
from utils import utils
import seaborn as sns
from eval import lbp

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

# plot histogram for intra/inter modal intra/inter class comparison


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return pairwise.cosine_similarity(a, b)

def dprime(nomatch, match):
    # nomatch = np.histogram(nomatch, bins=512)
    # match = np.histogram(match, bins=512)
    dprime = np.abs(np.mean(match) - np.mean(nomatch)) / np.sqrt(np.power(np.var(match), 2) + np.power(np.var(nomatch), 2))

    return round(dprime, 4)

def plot_hist(inter_class, intra_class, title, file__name, type='hist'):
    n_bins = 512
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 17, 'legend.fontsize': 15})    
    plt.figure()
    # plt.ylabel('Frequency')
    plt.xlabel('Cosine Similarity Score')
    plt.xticks(np.arange(-1, 1, 0.1))    
    plt.yticks([])
    dist1, dist2 = np.array(inter_class).ravel(), np.array(intra_class).ravel()
    if type == 'hist':
        fig, ax1 = plt.subplots()
        # ax1.hist(dist1, alpha=0.8, label='Inter-Subject', color='#1f77b4', bins=n_bins)
        sns.kdeplot(dist1, fill=True, alpha=0.5, label='Impostor', color='#1f77b4', ax=ax1) # 0.5
        ax1.tick_params(axis ='y', labelcolor='#1f77b4')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)

        ax2 = ax1.twinx()
        # ax2.hist(dist2, alpha=0.8, label='Intra-Subject', color='#ff7f0e', bins=n_bins)
        sns.kdeplot(dist2, fill=True, alpha=0.5, label='Genuine', color='#ff7f0e', ax=ax2) # 0.5
        ax2.tick_params(axis ='y', labelcolor='#ff7f0e')
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
        # ax1.set_ylabel(r"Inter-Class")
        # ax2.set_ylabel(r"Intra-Class")
    elif type == 'normal':
        # plt.plot(dist1, label='Inter-Subject', color='#1f77b4', bins=n_bins)        
        # plt.plot(dist2, label='Intra-Subject', color='#ff7f0e', bins=n_bins)
        sns.kdeplot(dist1, label='Inter-Subject', color='#1f77b4')
        sns.kdeplot(dist2, label='Intra-Subject', color='#ff7f0e')
    # plt.legend()
    plt.title("$d'$=" + str(dprime(dist1, dist2)))
    plt.savefig('./graphs/histogram/' + str(file__name) + '.svg', bbox_inches='tight')


def same_class_dist(fea, label):
    # same modalities, same class
    same_class_list = torch.tensor([])

    for i in torch.unique(label):
        # get index list where unique labels occur
        peri_indices = np.array(np.where(label == i)).ravel()
        sim_ = torch.Tensor(cosine_sim(fea[peri_indices], fea[peri_indices]).ravel())
        same_class_list = torch.cat((same_class_list, sim_), 0)
        
    return same_class_list.numpy()


# def diff_class_dist(fea, label):
#     # same modalities, different class
#     diff_class_list = torch.tensor([])

#     for i in torch.unique(label):
#         # get index list where unique labels occur
#         peri_indices = np.array(np.where(label == i)).ravel()
#         # indices that are not for label in question (i)
#         non_peri_indices = np.array(np.where(label != i)).ravel()
#         sim_ = torch.Tensor(cosine_sim(fea[peri_indices], fea[non_peri_indices]).ravel())
#         diff_class_list = torch.cat((diff_class_list, sim_), 0)

#     return diff_class_list.numpy()


def inter_model(peri_features, peri_label, face_features, face_label, cls):
    peri_features = peri_features.flatten(1)
    face_features = face_features.flatten(1)
    dist = torch.tensor([])

    for i in torch.unique(peri_label):
        peri_indices = np.array(np.where(peri_label == i)).ravel()
        non_peri_indices = np.array(np.where(peri_label != i)).ravel()

        # same classes, different modalities
        if cls == 'intra':
            sims = torch.Tensor(cosine_sim(peri_features[peri_indices], face_features[peri_indices]).ravel())
            dist = torch.cat((dist, sims), 0)
        # different classes, different modalities
        elif cls == 'inter':
            sims1 = torch.Tensor(cosine_sim(peri_features[peri_indices], face_features[non_peri_indices]).ravel())
            sims2 = torch.Tensor(cosine_sim(face_features[peri_indices], peri_features[non_peri_indices]).ravel())
            dist = torch.cat((dist, sims1), 0)
            dist = torch.cat((dist, sims2), 0)

    return dist.numpy()


def extract(model, data_path, modal):
    if modal == 'periocular':
        peri_flag = True
    else:
        peri_flag = False
    data_load, data_set = data_loader.gen_data(data_path, 'test', type=modal, aug='False')
    feat, labl = train.feature_extractor(model, data_load, device=device, peri_flag=peri_flag) 

    return feat, labl, data_load, data_set

if __name__ == '__main__':
    params = (8, 1)
    desc = lbp.LocalBinaryPatterns(8 , 2)
    lbp_inter_m_inter_c = []
    lbp_inter_m_intra_c = []

    dst_lst = [lbp_inter_m_inter_c, lbp_inter_m_intra_c]

    mf_peri = './data/test1_histogram/periocular/'
    mf_face = './data/test1_histogram/face/'

    peri_emb, peri_lbl = lbp.read_images(params, 'peri', mf_peri)
    face_emb, face_lbl = lbp.read_images(params, 'face', mf_face)  

    lbp_inter_m_inter_c = inter_model(peri_emb, peri_lbl, face_emb, face_lbl, cls='inter')
    lbp_inter_m_intra_c = inter_model(peri_emb, peri_lbl, face_emb, face_lbl, cls='intra')

    lbp_inter_class = []
    lbp_intra_class = []

    for lst_ in dst_lst:
        torch.save(lst_, ('./data/histogram/' + str(utils.retrieve_name(lst_)) + '.pt'))

    plot_hist(lbp_inter_m_inter_c, lbp_inter_m_intra_c, 'LBP', 'LBP', type='hist')