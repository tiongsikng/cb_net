import os, sys, glob, copy
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import time
import torch
import torch.utils.data
from torch.nn import functional as F
from data import data_loader
from sklearn.metrics import pairwise
from sklearn.model_selection import KFold
import network.cb_net as net
from network import load_model
from configs import config as config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

id_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_id_dict_f = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_id_dict_p = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
mm_id_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}

cmc_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_cmc_dict_f = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
cm_cmc_dict_p = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
mm_cmc_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
dset_name = ['Ethnic', 'Pubfig', 'FaceScrub', 'IMDb Wiki', 'AR']


def create_folder(method):
    lists = ['peri', 'face', 'cm', 'mm']
    boiler_path = './data/cmc/'
    for modal in lists:
        if not os.path.exists(os.path.join(boiler_path, method, modal)):
            os.makedirs(os.path.join(boiler_path, method, modal))


def get_avg(dict_list):
    total_ir = 0
    ir_list = []
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    if 'std' in dict_list.keys():
        del dict_list['std']
    for items in dict_list:
        total_ir += dict_list[items]
        ir_list.append(dict_list[items])
    dict_list['avg'] = total_ir/len(dict_list)
    dict_list['std'] = np.std(np.array(ir_list)) * 100

    return dict_list


# Intra-Modal Identification (Main)
def im_id_main(model, root_pth=config.evaluation['identification'], modal='periocular', peri_flag=True, device='cuda:0', proto_flag=False):
    print('Modal:', modal[:4])

    for datasets in dset_list:
        root_drt = root_pth + datasets + '/**'        
        modal_root = '/' + modal[:4] + '/'
        data_loaders = []
        data_sets = []
        acc = []        

        # data loader and datasets
        for directs in glob.glob(root_drt):
            base_nm = directs.split('\\')[-1]
            modal_base = directs + modal_root
            if not datasets in ['ethnic']:
                data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                data_loaders.append(data_load)
                data_sets.append(data_set)
        # *** ***

        if datasets == 'ethnic':
            ethnic_gal_data_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            ethnic_pr_data_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            acc = intramodal_id(model, ethnic_gal_data_load, ethnic_pr_data_load, device=device, peri_flag=peri_flag, proto_flag=proto_flag)

        else:
            # split data loaders into folds
            kf = KFold(n_splits=len(data_loaders))
            for probes, gallery in kf.split(data_loaders):
                for i in range(len(probes)):
                    peri_test_acc = intramodal_id(model, data_loaders[int(gallery)], data_loaders[probes[i]], 
                                                                                                device=device, peri_flag=peri_flag, proto_flag=proto_flag)
                    peri_test_acc = np.around(peri_test_acc, 4)
                    acc.append(peri_test_acc)

        # *** ***

        acc = np.around(np.mean(acc), 4)
        print(datasets, acc)
        id_dict[datasets] = acc

    return id_dict


# Cross-Modal Identification (Main)
def cm_id_main(model, root_pth=config.evaluation['identification'], face_model=None, peri_model=None, device='cuda:0', proto_flag=False):
    for datasets in dset_list:

        root_drt = root_pth + datasets + '/**'
        modal_root = ['/peri/', '/face/']
        path_lst = []
        acc_face_gal = []
        acc_peri_gal = []    

        # *** ***

        if datasets == 'ethnic':
            ethnic_face_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', 'face', aug='False')
            ethnic_peri_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', 'periocular', aug='False')
            inter_face_gal_acc_ethnic = crossmodal_id(model, ethnic_face_gal_load, ethnic_peri_pr_load, device=device, face_model=face_model, peri_model=peri_model, gallery='face', proto_flag=proto_flag)
            inter_face_gal_acc_ethnic = np.around(inter_face_gal_acc_ethnic, 4)
            acc_face_gal.append(inter_face_gal_acc_ethnic)

            ethnic_peri_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', 'periocular', aug='False')
            ethnic_face_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', 'face', aug='False')
            inter_peri_gal_acc_ethnic = crossmodal_id(model, ethnic_face_pr_load, ethnic_peri_gal_load, device=device, face_model=face_model, peri_model=peri_model, gallery='peri', proto_flag=proto_flag)
            inter_peri_gal_acc_ethnic = np.around(inter_peri_gal_acc_ethnic, 4)
            acc_peri_gal.append(inter_peri_gal_acc_ethnic)

        else:
            # data loader and datasets
            for directs in glob.glob(root_drt):
                base_nm = directs.split('\\')[-1]
                if not directs.split('/')[-1] == 'gallery':
                    path_lst.append(directs)
                else:
                    gallery_path = directs      

            fold = 0
            kf = KFold(n_splits=len(path_lst))
            for probes in path_lst:
                fold += 1
                # print(path_lst[int(probes[i])] + modal_root[0], path_lst[int(gallery)] + modal_root[1])
                peri_probe_load, peri_dataset = data_loader.gen_data((probes + modal_root[0]), 'test', 'periocular', aug='False')
                face_gal_load, face_dataset = data_loader.gen_data((gallery_path + modal_root[1]), 'test', 'face', aug='False')
                cm_face_gal_acc = crossmodal_id(model, face_gal_load, peri_probe_load, device=device, proto_flag=False, face_model=face_model, peri_model=peri_model, gallery='face')
                cm_face_gal_acc = np.around(cm_face_gal_acc, 4)
                acc_face_gal.append(cm_face_gal_acc)

                peri_gal_load, peri_dataset = data_loader.gen_data((gallery_path + modal_root[0]), 'test', 'periocular', aug='False')
                face_probe_load, face_dataset = data_loader.gen_data((probes + modal_root[1]), 'test', 'face', aug='False')
                cm_peri_gal_acc = crossmodal_id(model, face_probe_load, peri_gal_load, device=device, proto_flag=False, face_model=face_model, peri_model=peri_model, gallery='peri')
                cm_peri_gal_acc = np.around(cm_peri_gal_acc, 4)
                acc_peri_gal.append(cm_peri_gal_acc)

        # *** ***

        acc_peri_gal = np.around(np.mean(acc_peri_gal), 4)
        acc_face_gal = np.around(np.mean(acc_face_gal), 4)        
        print('Peri Gallery:', datasets, acc_peri_gal)
        print('Face Gallery:', datasets, acc_face_gal) 
        cm_id_dict_p[datasets] = acc_peri_gal       
        cm_id_dict_f[datasets] = acc_face_gal        

    return cm_id_dict_f, cm_id_dict_p


# Multimodal Identification (Main)
def mm_id_main(model, root_pth='./data/', face_model=None, peri_model=None, mode='concat', device='cuda:0', proto_flag=False):
    for datasets in dset_list:
        root_drt = root_pth + datasets + '/**'
        modal_root = ['/peri/', '/face/']
        path_lst = []
        data_loaders = []
        acc = []                

        # *** ***

        if datasets == 'ethnic':
            ethnic_face_gal_data_load, ethnic_face_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', 'face', aug='False')
            ethnic_face_pr_data_load, ethnic_face_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', 'face', aug='False')
            ethnic_peri_gal_data_load, ethnic_peri_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', 'periocular', aug='False')            
            ethnic_peri_pr_data_load, ethnic_peri_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', 'periocular', aug='False')        
            acc = multimodal_id(model, ethnic_face_gal_data_load, ethnic_peri_gal_data_load, ethnic_face_pr_data_load, ethnic_peri_pr_data_load, 
                                    device=device, proto_flag=False, face_model=face_model, peri_model=peri_model, mode=mode)
        else:
            # data loader and datasets
            for directs in glob.glob(root_drt):
                base_nm = directs.split('\\')[-1]
                path_lst.append(directs)        

            # split data loaders into folds
            fold = 0
            kf = KFold(n_splits=len(path_lst))
            for probes, gallery in kf.split(path_lst):
                fold += 1
                for i in range(len(probes)):
                    # print(path_lst[int(probes[i])] + modal_root[0], path_lst[int(gallery)] + modal_root[1])
                    face_data_load_gal, face_dataset_gal = data_loader.gen_data((path_lst[int(gallery)] + modal_root[1]), 'test', 'face', aug='False')
                    face_data_load_pr, face_dataset_pr = data_loader.gen_data((path_lst[int(probes[i])] + modal_root[1]), 'test', 'face', aug='False')
                    peri_data_load_gal, peri_dataset_gal = data_loader.gen_data((path_lst[int(gallery)] + modal_root[0]), 'test', 'periocular', aug='False')                    
                    peri_data_load_pr, peri_dataset_pr = data_loader.gen_data((path_lst[int(probes[i])] + modal_root[0]), 'test', 'periocular', aug='False')
                    
                    mm_test_acc = multimodal_id(model, face_data_load_gal, peri_data_load_gal, face_data_load_pr, peri_data_load_pr, 
                                                            device=device, proto_flag=False, face_model=face_model, peri_model=peri_model, mode=mode)
                    mm_test_acc = np.around(mm_test_acc, 4)
                    acc.append(mm_test_acc)
                #     print(i, mm_test_acc)
                # print("Fold:", fold)

        # *** ***

        acc = np.around(np.mean(np.array(acc)), 4)
        print(datasets, acc)
        mm_id_dict[datasets] = acc

    return mm_id_dict


# Intra-Modal Identification Function
def intramodal_id(model, loader_gallery, loader_test, device='cuda:0', peri_flag=False, proto_flag=False):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False
        
    # ***** *****
    
    # Extract gallery features w.r.t. pre-learned model
    gallery_fea = torch.tensor([])
    gallery_label = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_gallery):

            x = x.to(device)
            x = model(x, peri_flag=peri_flag)

            gallery_fea = torch.cat((gallery_fea, x.detach().cpu()), 0)
            gallery_label = torch.cat((gallery_label, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(gallery_fea.size()[0] == gallery_label.size()[0])
    
    del loader_gallery
    time.sleep(0.0001)
    
    # ***** *****
    
    # Extract test features w.r.t. pre-learned model
    test_fea = torch.tensor([])
    test_label = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_test):

            x = x.to(device)
            x = model(x, peri_flag=peri_flag)

            test_fea = torch.cat((test_fea, x.detach().cpu()), 0)
            test_label = torch.cat((test_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    # print('Test Set Capacity\t: ', test_fea.size())
    assert(test_fea.size()[0] == test_label.size()[0])
    
    del loader_test
    time.sleep(0.0001)

    # ***** *****
    # prototyping for gallery
    if proto_flag is True:
        gal_lbl_proto = torch.tensor([], dtype=torch.int64)
        gal_fea_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(gallery_label):
            # append unique labels to tensor list
            gal_lbl_proto = torch.cat((gal_lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            indices = np.where(gallery_label == i)
            gal_feats = torch.tensor([])

            # from index list, append features into temporary gal_feats list
            for j in indices:
                gal_feats = torch.cat((gal_feats, gallery_fea[j].detach().cpu()), 0)
            # print(gal_feats.shape)
            # get mean of full gal_feats list, and then unsqueeze to append the average prototype value into gal_fea_proto
            proto_mean = torch.unsqueeze(torch.mean(gal_feats, 0), 0)
            proto_mean = F.normalize(proto_mean, p=2, dim=1)
            gal_fea_proto = torch.cat((gal_fea_proto, proto_mean.detach().cpu()), 0)
    
        gallery_fea, gallery_label = gal_fea_proto, gal_lbl_proto

    # Calculate gallery_acc and test_acc
    gallery_label = np.reshape(np.array(gallery_label), -1)
    test_label = np.reshape(np.array(test_label), -1)
    
    test_dist = pairwise.cosine_similarity(gallery_fea, test_fea)
    test_pred = np.argmax(test_dist, 0)
    test_pred = gallery_label[test_pred]
    test_acc = sum(test_label == test_pred) / test_label.shape[0]

    # torch.cuda.empty_cache()
    # time.sleep(0.0001)
    
    del model
    time.sleep(0.0001)
    
    return test_acc


# Cross-Modal Identification Function
def crossmodal_id(model, face_loader, peri_loader, device='cuda:0', face_model=None, peri_model=None, gallery='face', proto_flag=False):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False

    # ***** *****

    # Extract face features w.r.t. pre-learned model
    face_fea = torch.tensor([])
    face_label = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag=False)
            else:
                x = model(x, peri_flag=False)

            face_fea = torch.cat((face_fea, x.detach().cpu()), 0)
            face_label = torch.cat((face_label, y))
            
            del x, y
            time.sleep(0.0001)
    
    # print('Test Set Capacity\t: ', test_fea.size())
    assert(face_fea.size()[0] == face_label.size()[0])
    
    del face_loader
    time.sleep(0.0001)

    # *****    
    
    # Extract periocular features w.r.t. pre-learned model
    peri_fea = torch.tensor([])
    peri_label = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(peri_loader):

            x = x.to(device)
            if not peri_model is None:
                peri_model = peri_model.eval().to(device)
                x = peri_model(x, peri_flag=True)
            else:
                x = model(x, peri_flag=True)

            peri_fea = torch.cat((peri_fea, x.detach().cpu()), 0)
            peri_label = torch.cat((peri_label, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(peri_fea.size()[0] == peri_label.size()[0])
    
    del peri_loader
    time.sleep(0.0001)
    
    # ***** *****

    # perform checking
    if gallery == 'face':
        gal_fea, gal_label = face_fea, face_label
        probe_fea, probe_label = peri_fea, peri_label
    elif gallery == 'peri':
        gal_fea, gal_label = peri_fea, peri_label
        probe_fea, probe_label = face_fea, face_label

    # normalize features
    gal_fea = F.normalize(gal_fea, p=2, dim=1)
    probe_fea = F.normalize(probe_fea, p=2, dim=1)

    # Calculate gallery_acc and test_acc
    gal_label = np.reshape(np.array(gal_label), -1)
    probe_label = np.reshape(np.array(probe_label), -1)    
    
    probe_dist = pairwise.cosine_similarity(gal_fea, probe_fea)
    probe_pred = np.argmax(probe_dist, 0)
    probe_pred = gal_label[probe_pred]
    probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]
    
    del model
    time.sleep(0.0001)
    
    return probe_acc


# Multimodal Identification Function
def multimodal_id(model, face_loader_gal, peri_loader_gal, face_loader_probe, peri_loader_probe, device='cuda:0', proto_flag=False, face_model=None, peri_model=None, mode='concat'):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False

    # ***** *****

    gal_fea = torch.tensor([])
    gal_label = torch.tensor([], dtype=torch.int64)
    probe_fea = torch.tensor([])
    probe_label = torch.tensor([], dtype=torch.int64)

    # ***** *****

    # GALLERY
    # Extract face features w.r.t. pre-learned model
    face_fea_gal = torch.tensor([])
    face_label_gal = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader_gal):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag=False)
            else:
                x = model(x, peri_flag=False)

            face_fea_gal = torch.cat((face_fea_gal, x.detach().cpu()), 0)
            face_label_gal = torch.cat((face_label_gal, y))
            
            del x, y
            time.sleep(0.0001)

    assert(face_fea_gal.size()[0] == face_label_gal.size()[0])
    
    del face_loader_gal
    time.sleep(0.0001)

    # *****    
    
    # Extract periocular features w.r.t. pre-learned model
    peri_fea_gal = torch.tensor([])
    peri_label_gal = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(peri_loader_gal):

            x = x.to(device)
            if not peri_model is None:
                peri_model = peri_model.eval().to(device)
                x = peri_model(x, peri_flag=True)
            else:
                x = model(x, peri_flag=True)

            peri_fea_gal = torch.cat((peri_fea_gal, x.detach().cpu()), 0)
            peri_label_gal = torch.cat((peri_label_gal, y))
            
            del x, y
            time.sleep(0.0001)

    assert(peri_fea_gal.size()[0] == peri_label_gal.size()[0])
    
    del peri_loader_gal
    time.sleep(0.0001)
    
    # ***** *****
    
    # PROBE
    # Extract face features w.r.t. pre-learned model
    face_fea_probe = torch.tensor([])
    face_label_probe = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader_probe):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag=False)
            else:
                x = model(x, peri_flag=False)

            face_fea_probe = torch.cat((face_fea_probe, x.detach().cpu()), 0)
            face_label_probe = torch.cat((face_label_probe, y))
            
            del x, y
            time.sleep(0.0001)
    
    assert(face_fea_probe.size()[0] == face_label_probe.size()[0])
    
    del face_loader_probe
    time.sleep(0.0001)

    # *****    
    
    # Extract periocular features w.r.t. pre-learned model
    peri_fea_probe = torch.tensor([])
    peri_label_probe = torch.tensor([], dtype=torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(peri_loader_probe):

            x = x.to(device)
            if not peri_model is None:
                peri_model = peri_model.eval().to(device)
                x = peri_model(x, peri_flag=True)
            else:
                x = model(x, peri_flag=True)

            peri_fea_probe = torch.cat((peri_fea_probe, x.detach().cpu()), 0)
            peri_label_probe = torch.cat((peri_label_probe, y))
            
            del x, y
            time.sleep(0.0001)

    # print('Gallery Set Capacity\t: ', gallery_fea.size())
    assert(peri_fea_probe.size()[0] == peri_label_probe.size()[0])
    
    del peri_loader_probe
    time.sleep(0.0001)
    
    # ***** *****
    
    # for prototyping
    if proto_flag is True:
        print("WITH Prototyping on periocular and face galleries.")
        # periocular
        peri_lbl_proto = torch.tensor([], dtype=torch.int64)
        peri_fea_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(peri_label_gal):
            # append unique labels to tensor list
            peri_lbl_proto = torch.cat((peri_lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            peri_indices = np.where(peri_label_gal == i)
            peri_feats = torch.tensor([])

            # from index list, append features into temporary peri_feats list
            for j in peri_indices:
                peri_feats = torch.cat((peri_feats, peri_fea_gal[j].detach().cpu()), 0)
            peri_proto_mean = torch.unsqueeze(torch.mean(peri_feats, 0), 0)
            peri_proto_mean = F.normalize(peri_proto_mean, p=2, dim=1)
            peri_fea_proto = torch.cat((peri_fea_proto, peri_proto_mean.detach().cpu()), 0)

        # finally, set periocular feature and label to prototyped ones
        peri_fea_gal, peri_label_gal = peri_fea_proto, peri_lbl_proto

        #### **** ####

        # face
        face_lbl_proto = torch.tensor([], dtype=torch.int64)
        face_fea_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(face_label_gal):
            # append unique labels to tensor list
            face_lbl_proto = torch.cat((face_lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            face_indices = np.where(face_label_gal == i)
            face_feats = torch.tensor([])

            # from index list, append features into temporary face_feats list
            for j in face_indices:
                face_feats = torch.cat((face_feats, face_fea_gal[j].detach().cpu()), 0)
            face_proto_mean = torch.unsqueeze(torch.mean(face_feats, 0), 0)
            face_proto_mean = F.normalize(face_proto_mean, p=2, dim=1)
            face_fea_proto = torch.cat((face_fea_proto, face_proto_mean.detach().cpu()), 0)

        # finally, set face feature and label to prototyped ones
        face_fea_gal, face_label_gal = face_fea_proto, face_lbl_proto

    # ***** *****
    # concatenate (face|peri) for gallery and probe
    face_fea_gal, face_fea_probe = face_fea_gal.detach().cpu(), face_fea_probe.detach().cpu()
    peri_fea_gal, peri_fea_probe = peri_fea_gal.detach().cpu(), peri_fea_probe.detach().cpu()
    gal_label = face_label_gal
    probe_label = face_label_probe

    if mode == 'concat':
        gal_fea = torch.cat((gal_fea, face_fea_gal.detach().cpu()), 1)
        gal_fea = torch.cat((gal_fea, peri_fea_gal.detach().cpu()), 1)
        gal_label = face_label_gal

        gal_fea = F.normalize(gal_fea, p=2, dim=1)

        probe_fea = torch.cat((probe_fea, face_fea_probe.detach().cpu()), 1)
        probe_fea = torch.cat((probe_fea, peri_fea_probe.detach().cpu()), 1)
        probe_label = face_label_probe
        
        probe_fea = F.normalize(probe_fea, p=2, dim=1)
    
    elif mode == 'mean' or mode == 'max':
        face_fea_gal = torch.unsqueeze( face_fea_gal.detach().cpu(), 0)
        peri_fea_gal = torch.unsqueeze( peri_fea_gal.detach().cpu(), 0)
        
        gal_fea = torch.cat((face_fea_gal, peri_fea_gal), 0) # make sure 2 x N x d
        if mode == 'mean':
            gal_fea = torch.mean(gal_fea, 0) # make sure N x d
        elif mode == 'max':
            gal_fea = torch.max(gal_fea, 0)[0] # make sure N x d
        gal_fea = F.normalize(gal_fea, p=2, dim=1)
        
        face_fea_probe = torch.unsqueeze( face_fea_probe.detach().cpu(), 0)
        peri_fea_probe = torch.unsqueeze( peri_fea_probe.detach().cpu(), 0)
        
        probe_fea = torch.cat((face_fea_probe, peri_fea_probe), 0) # make sure 2 x N x d
        if mode == 'mean':
            probe_fea = torch.mean(probe_fea, 0) # make sure N x d
        elif mode == 'max':
            probe_fea = torch.max(probe_fea, 0)[0] # make sure N x d
        probe_fea = F.normalize(probe_fea, p=2, dim=1)
    
    elif mode == 'score':
        face_dist = pairwise.cosine_similarity(face_fea_gal, face_fea_probe)
        peri_dist = pairwise.cosine_similarity(peri_fea_gal, peri_fea_probe)
        # face_dist = torch.matmul(face_fea_probe, face_fea_gal.t())
        # peri_dist = torch.matmul(peri_fea_probe, peri_fea_gal.t())
        face_dist = torch.unsqueeze(torch.Tensor(face_dist).detach().cpu(), 0)
        peri_dist = torch.unsqueeze(torch.Tensor(peri_dist).detach().cpu(), 0)

        distances = torch.cat((face_dist, peri_dist), 0)
        prediction_score = torch.mean(distances, 0)
        # prediction_score = F.normalize(prediction_score, p=2, dim=1)
        
        # scores_mean = (face_dist + peri_dist) / 2
        # scores_mean = np.array(F.normalize(torch.Tensor(scores_mean), p=2, dim=1))

        probe_pred = np.argmax(prediction_score.numpy(), 0)
        probe_pred = gal_label[probe_pred]
        probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]
        return 1, probe_acc
    
    # ***** *****

    # Calculate gallery_acc and probe_acc
    gal_label = np.reshape(np.array(gal_label), -1)
    probe_label = np.reshape(np.array(probe_label), -1)

    probe_dist = pairwise.cosine_similarity(gal_fea, probe_fea)
    probe_pred = np.argmax(probe_dist, 0)
    probe_pred = gal_label[probe_pred]
    probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]

    torch.cuda.empty_cache()
    # time.sleep(0.0001)
    
    del model
    time.sleep(0.0001)
    
    return probe_acc


def feature_extractor(model, data_loader, device='cuda:0', peri_flag=False, proto_flag=False):    
    emb = torch.tensor([])
    lbl = torch.tensor([], dtype=torch.int64)

    model = model.eval().to(device)
    
    with torch.no_grad():        
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x = model(x, peri_flag=peri_flag)

            emb = torch.cat((emb, x.detach().cpu()), 0)
            lbl = torch.cat((lbl, y))
            
            del x, y
            time.sleep(0.0001)

    if proto_flag is True:
        lbl_proto = torch.tensor([], dtype=torch.int64)
        emb_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(lbl):
            # append unique labels to tensor list
            lbl_proto = torch.cat((lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            indices = np.where(lbl == i)
            feats = torch.tensor([])

            # from index list, append features into temporary feats list
            for j in indices:
                feats = torch.cat((feats, emb[j].detach().cpu()), 0)
            # print(feats.shape)
            # get mean of full feats list, and then unsqueeze to append the average prototype value into gal_fea_proto
            proto_mean = torch.unsqueeze(torch.mean(feats, 0), 0)
            proto_mean = F.normalize(proto_mean, p=2, dim=1)
            emb_proto = torch.cat((emb_proto, proto_mean.detach().cpu()), 0)
    
        emb, lbl = emb_proto, lbl_proto

    # print('Set Capacity\t: ', emb.size())
    assert(emb.size()[0] == lbl.size()[0])
    
    del data_loader
    time.sleep(0.0001)

    del model
    
    return emb, lbl


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

    return cmc.numpy()


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

    return cmc.numpy()


def im_cmc_extractor(model, root_pth='./data/', modal='periocular', peri_flag=True, device='cuda:0', rank=10):
    for datasets in dset_list:
        cmc_lst = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'        
        modal_root = '/' + modal[:4] + '/'
        data_loaders = []
        data_sets = []
        probe_data_loaders = []
        probe_data_sets = []               

        # data loader and datasets
        if not datasets in ['ethnic', 'ytf']:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('\\')[-1]
                modal_base = directs + modal_root
                if modal_base.split('/')[-3] != 'gallery':
                    data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                    # probe_data_loaders.append(data_load)
                    # probe_data_sets.append(data_set)
                    data_loaders.append(data_load)
                    data_sets.append(data_set)
                else:
                    data_load, data_set = data_loader.gen_data(modal_base, 'test', type=modal, aug='False')
                    # gallery_data_loaders = data_load
                    # gallery_data_sets = data_set
                    data_loaders.append(data_load)
                    data_sets.append(data_set)               
        
        # *** ***
        if datasets == 'ethnic':
            ethnic_gal_data_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            ethnic_pr_data_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/' + modal[:4] + '/'), 'test', type=modal, aug='False')
            ethnic_fea_gal, ethnic_lbl_gal = feature_extractor(model, ethnic_gal_data_load, device=device, peri_flag=peri_flag)
            ethnic_fea_pr, ethnic_lbl_pr = feature_extractor(model, ethnic_pr_data_load, device=device, peri_flag=peri_flag)            
            ethnic_lbl_pr, ethnic_lbl_gal = F.one_hot(ethnic_lbl_pr), F.one_hot(ethnic_lbl_gal)
            cmc = calculate_cmc(ethnic_fea_gal, ethnic_fea_pr, ethnic_lbl_gal, ethnic_lbl_pr, last_rank=rank)

        else:
            # split data loaders into folds
            kf = KFold(n_splits=len(data_loaders))
            for probes, gallery in kf.split(data_loaders):            
                for i in range(len(probes)):
                    peri_fea_gal, peri_lbl_gal = feature_extractor(model, data_loaders[int(gallery)], device=device, peri_flag=peri_flag)
                    peri_fea_pr, peri_lbl_pr = feature_extractor(model, data_loaders[int(probes[i])], device=device, peri_flag=peri_flag)
                    peri_lbl_pr, peri_lbl_gal = F.one_hot(peri_lbl_pr), F.one_hot(peri_lbl_gal)
                    cmc = calculate_cmc(peri_fea_gal, peri_fea_pr, peri_lbl_gal, peri_lbl_pr, last_rank=rank)
                    cmc_lst = np.append(cmc_lst, np.array([cmc]), axis=0)                
                cmc = np.mean(cmc_lst, axis=0)
        cmc_dict[datasets] = cmc
        print(datasets, cmc)
        # print(cmc_dict)

    return cmc_dict


def cm_cmc_extractor(model, root_pth='./data/', facenet=None, perinet=None, device='cuda:0', rank=10):
    if facenet is None and perinet is None:
        facenet = model
        perinet = model

    for datasets in dset_list:
        cmc_lst_f = np.empty((0, rank), int)
        cmc_lst_p = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'     
        peri_data_loaders = []
        peri_data_sets = []     
        face_data_loaders = []
        face_data_sets = []            

        # data loader and datasets
        if not datasets in ['ethnic', 'ytf']:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('/')[-1]
                if base_nm == 'gallery':
                    peri_data_load_gal, peri_data_set_gal = data_loader.gen_data(directs + '/peri/', 'test', type='periocular', aug='False')
                    face_data_load_gal, face_data_set_gal = data_loader.gen_data(directs + '/face/', 'test', type='face', aug='False')
                else:
                    peri_data_load, peri_data_set = data_loader.gen_data(directs + '/peri/', 'test', type='periocular', aug='False')
                    face_data_load, face_data_set = data_loader.gen_data(directs + '/face/', 'test', type='face', aug='False')
                    peri_data_loaders.append(peri_data_load)
                    peri_data_sets.append(peri_data_set)
                    face_data_loaders.append(face_data_load)
                    face_data_sets.append(face_data_set)
        # print(datasets)
        # *** ***
        if datasets == 'ethnic':
            p_ethnic_gal_data_load, p_ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', type='periocular', aug='False')
            p_ethnic_pr_data_load, p_ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', type='periocular', aug='False')
            p_ethnic_fea_gal, p_ethnic_lbl_gal = feature_extractor(perinet, p_ethnic_gal_data_load, device=device, peri_flag=True)
            p_ethnic_fea_pr, p_ethnic_lbl_pr = feature_extractor(perinet, p_ethnic_pr_data_load, device=device, peri_flag=True)            
            p_ethnic_lbl_pr, p_ethnic_lbl_gal = F.one_hot(p_ethnic_lbl_pr), F.one_hot(p_ethnic_lbl_gal)

            f_ethnic_gal_data_load, f_ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', type='face', aug='False')
            f_ethnic_pr_data_load, f_ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', type='face', aug='False')
            f_ethnic_fea_gal, f_ethnic_lbl_gal = feature_extractor(facenet, f_ethnic_gal_data_load, device=device, peri_flag=False)
            f_ethnic_fea_pr, f_ethnic_lbl_pr = feature_extractor(facenet, f_ethnic_pr_data_load, device=device, peri_flag=False)            
            f_ethnic_lbl_pr, f_ethnic_lbl_gal = F.one_hot(f_ethnic_lbl_pr), F.one_hot(f_ethnic_lbl_gal)

            cmc_f = calculate_cmc(f_ethnic_fea_gal, p_ethnic_fea_pr, f_ethnic_lbl_gal, p_ethnic_lbl_pr, last_rank=rank)
            cmc_p = calculate_cmc(p_ethnic_fea_gal, f_ethnic_fea_pr, p_ethnic_lbl_gal, f_ethnic_lbl_pr, last_rank=rank)

        else:            
            for probes in peri_data_loaders:                
                face_fea_gal, face_lbl_gal = feature_extractor(perinet, face_data_load_gal, device=device, peri_flag=False)
                peri_fea_pr, peri_lbl_pr = feature_extractor(perinet, probes, device=device, peri_flag=True)
                peri_lbl_pr, face_lbl_gal = F.one_hot(peri_lbl_pr), F.one_hot(face_lbl_gal)

                cmc_f = calculate_cmc(face_fea_gal, peri_fea_pr, face_lbl_gal, peri_lbl_pr, last_rank=rank)
                cmc_lst_f = np.append(cmc_lst_f, np.array([cmc_f]), axis=0)

            for probes in face_data_loaders:                
                peri_fea_gal, peri_lbl_gal = feature_extractor(perinet, peri_data_load_gal, device=device, peri_flag=True)
                face_fea_pr, face_lbl_pr = feature_extractor(perinet, probes, device=device, peri_flag=False)
                face_lbl_pr, peri_lbl_gal = F.one_hot(face_lbl_pr), F.one_hot(peri_lbl_gal)

                cmc_p = calculate_cmc(peri_fea_gal, face_fea_pr, peri_lbl_gal, face_lbl_pr, last_rank=rank)
                cmc_lst_p = np.append(cmc_lst_p, np.array([cmc_p]), axis=0)
                
            cmc_f = np.mean(cmc_lst_f, axis=0)
            cmc_p = np.mean(cmc_lst_p, axis=0)

        cm_cmc_dict_p[datasets] = cmc_p
        cm_cmc_dict_f[datasets] = cmc_f
        print(datasets)
        print('Peri Gallery:', cmc_p)        
        print('Face Gallery:', cmc_f)

    return cm_cmc_dict_f, cm_cmc_dict_p


def mm_cmc_extractor(model, root_pth='./data/', facenet=None, perinet=None, device='cuda:0', rank=10, mode='concat'):
    if facenet is None and perinet is None:
        facenet = model
        perinet = model

    for datasets in dset_list:
        cmc_lst = np.empty((0, rank), int)
        root_drt = root_pth + datasets + '/**'     
        peri_data_loaders = []
        peri_data_sets = []     
        face_data_loaders = []
        face_data_sets = []            

        # data loader and datasets
        if not datasets in ['ethnic', 'ytf']:
            for directs in glob.glob(root_drt):
                base_nm = directs.split('\\')[-1]
                peri_data_load, peri_data_set = data_loader.gen_data(directs + '/peri/', 'test', type='periocular', aug='False')
                face_data_load, face_data_set = data_loader.gen_data(directs + '/face/', 'test', type='face', aug='False')
                peri_data_loaders.append(peri_data_load)
                peri_data_sets.append(peri_data_set)
                face_data_loaders.append(face_data_load)
                face_data_sets.append(face_data_set)
        # print(datasets)
        # *** ***
        if datasets == 'ethnic':
            ethnic_fea_gal = torch.Tensor([])
            ethnic_fea_pr = torch.Tensor([])

            p_ethnic_gal_data_load, p_ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', type='periocular', aug='False')
            p_ethnic_pr_data_load, p_ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', type='periocular', aug='False')
            p_ethnic_fea_gal, p_ethnic_lbl_gal = feature_extractor(perinet, p_ethnic_gal_data_load, device=device, peri_flag=True)
            p_ethnic_fea_pr, p_ethnic_lbl_pr = feature_extractor(perinet, p_ethnic_pr_data_load, device=device, peri_flag=True)            
            p_ethnic_lbl_pr, p_ethnic_lbl_gal = F.one_hot(p_ethnic_lbl_pr), F.one_hot(p_ethnic_lbl_gal)

            f_ethnic_gal_data_load, f_ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', type='face', aug='False')
            f_ethnic_pr_data_load, f_ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', type='face', aug='False')
            f_ethnic_fea_gal, f_ethnic_lbl_gal = feature_extractor(facenet, f_ethnic_gal_data_load, device=device, peri_flag=False)
            f_ethnic_fea_pr, f_ethnic_lbl_pr = feature_extractor(facenet, f_ethnic_pr_data_load, device=device, peri_flag=False)            
            f_ethnic_lbl_pr, f_ethnic_lbl_gal = F.one_hot(f_ethnic_lbl_pr), F.one_hot(f_ethnic_lbl_gal)

            cmc = calculate_mm_cmc(p_ethnic_fea_gal, p_ethnic_fea_pr, p_ethnic_lbl_gal, p_ethnic_lbl_pr, f_ethnic_fea_gal, f_ethnic_fea_pr, f_ethnic_lbl_gal, f_ethnic_lbl_pr, 
                                            last_rank=rank, mode = mode)

        else:            
            fea_gal = torch.Tensor([])
            fea_pr = torch.Tensor([])
            # split data loaders into folds
            kf = KFold(n_splits=len(peri_data_loaders))
            for probes, gallery in kf.split(peri_data_loaders):                
                for i in range(len(probes)):
                    peri_fea_gal, peri_lbl_gal = feature_extractor(perinet, peri_data_loaders[int(gallery)], device=device, peri_flag=True)
                    peri_fea_pr, peri_lbl_pr = feature_extractor(perinet, peri_data_loaders[probes[i]], device=device, peri_flag=True)
                    peri_lbl_pr, peri_lbl_gal = F.one_hot(peri_lbl_pr), F.one_hot(peri_lbl_gal)

                    face_fea_gal, face_lbl_gal = feature_extractor(facenet, face_data_loaders[int(gallery)], device=device, peri_flag=False)
                    face_fea_pr, face_lbl_pr = feature_extractor(facenet, face_data_loaders[probes[i]], device=device, peri_flag=False)
                    face_lbl_pr, face_lbl_gal = F.one_hot(face_lbl_pr), F.one_hot(face_lbl_gal)

                    cmc = calculate_mm_cmc(peri_fea_gal, peri_fea_pr, peri_lbl_gal, peri_lbl_pr, face_fea_gal, face_fea_pr, face_lbl_gal, face_lbl_pr, last_rank=rank, mode=mode)
                    cmc_lst = np.append(cmc_lst, np.array([cmc]), axis=0)
                
            cmc = np.mean(cmc_lst, axis=0)

        mm_cmc_dict[datasets] = cmc
        print(datasets, cmc)

    return mm_cmc_dict


if __name__ == '__main__':
    method = 'cb_net'
    rank = 10 # CMC - rank > 1 (graph) or identification - rank = 1 (values)
    mm_modes_list = ['concat', 'mean', 'max', 'score']
    if rank > 1:
        create_folder(method)
    embd_dim = 512
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_model_path = './models/best_model/CB_Net.pth'
    model = net.CB_Net(embedding_size=embd_dim, do_prob=0.0).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device=device)

    load_face_model_path = './models/best_model/CB_Net_Face_Baseline.pth'
    face_model = net.CB_Net(embedding_size=embd_dim, do_prob=0.0).eval().to(device)
    face_model = load_model.load_pretrained_network(face_model, load_face_model_path, device=device)

    load_peri_model_path = './models/best_model/CB_Net_Peri_Baseline.pth'
    peri_model = net.CB_Net(embedding_size=embd_dim, do_prob=0.0).eval().to(device)
    peri_model = load_model.load_pretrained_network(peri_model, load_peri_model_path, device=device)

    #### Compute CMC Values
    peri_cmc_dict = im_cmc_extractor(model, root_pth=config.evaluation['identification'], modal='periocular', peri_flag=True, device=device, rank=rank)
    peri_cmc_dict = get_avg(peri_cmc_dict)
    if rank > 1:
        peri_cmc_dict = copy.deepcopy(peri_cmc_dict)
        torch.save(peri_cmc_dict, './data/cmc/' + str(method) + '/peri/peri_cmc_dict.pt')
    print('Periocular: \n', peri_cmc_dict)
    print('Average (Periocular): \n', peri_cmc_dict['avg'], '±', peri_cmc_dict['std']) 

    face_cmc_dict = im_cmc_extractor(model, root_pth=config.evaluation['identification'], modal='face', peri_flag=False, device=device, rank=rank)
    face_cmc_dict = get_avg(face_cmc_dict)
    if rank > 1:
        face_cmc_dict = copy.deepcopy(face_cmc_dict)
        torch.save(face_cmc_dict, './data/cmc/' + str(method) + '/face/face_cmc_dict.pt') 
    print('Face: \n', face_cmc_dict)    
    print('Average (Face): \n', face_cmc_dict['avg'], '±', face_cmc_dict['std'])     

    cm_cmc_dict_f, cm_cmc_dict_p = cm_cmc_extractor(model, facenet=None, perinet=None, root_pth=config.evaluation['identification'], device=device, rank=rank)
    cm_cmc_dict_f = get_avg(cm_cmc_dict_f)
    cm_cmc_dict_p = get_avg(cm_cmc_dict_p)
    if rank > 1:
        cm_cmc_dict_f = copy.deepcopy(cm_cmc_dict_f)
        cm_cmc_dict_p = copy.deepcopy(cm_cmc_dict_p)   
        torch.save(cm_cmc_dict_f, './data/cmc/' + str(method) + '/cm/cm_cmc_dict_f.pt')
        torch.save(cm_cmc_dict_p, './data/cmc/' + str(method) + '/cm/cm_cmc_dict_p.pt')
    print('Cross-Modal: \n', cm_cmc_dict_f, cm_cmc_dict_p)
    print('Average (Cross-Modal Periocular): \n', cm_cmc_dict_p['avg'], '±', cm_cmc_dict_p['std']) 
    print('Average (Cross-Modal Face): \n', cm_cmc_dict_f['avg'], '±', cm_cmc_dict_f['std']) 
    
    for mm_mode in mm_modes_list:
        mm_cmc_dict = mm_cmc_extractor(model, facenet=None, perinet=None, root_pth=config.evaluation['identification'], device=device, rank=rank, mode=mm_mode)
        mm_cmc_dict = get_avg(mm_cmc_dict)
        if rank > 1:
            mm_cmc_dict = copy.deepcopy(mm_cmc_dict)
            torch.save(mm_cmc_dict, './data/cmc/' + str(method) + '/mm/mm_cmc_dict_' + str(mm_mode) + '.pt')
        print('Multimodal: \n', mm_cmc_dict)
        print('Average (Periocular+Face): \n', mm_cmc_dict['avg'], '±', mm_cmc_dict['std']) 