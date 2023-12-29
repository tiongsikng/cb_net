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
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar'] 


def get_avg(dict_list):
    total_ir = 0
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    for items in dict_list:
        total_ir += dict_list[items]
    dict_list['avg'] = total_ir/len(dict_list)

    return dict_list


# Identification (Main)
def kfold_identification(model, root_pth=config.evaluation['identification'], modal='periocular', peri_flag = True, device = 'cuda:0', proto_flag = False):
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
            _, acc = validate_identification(model, ethnic_gal_data_load, ethnic_pr_data_load, device = device, peri_flag = peri_flag, proto_flag = proto_flag)

        else:
            # split data loaders into folds
            kf = KFold(n_splits=len(data_loaders))
            for probes, gallery in kf.split(data_loaders):
                for i in range(len(probes)):
                    peri_acc1, peri_test_acc = validate_identification(model, data_loaders[int(gallery)], data_loaders[probes[i]], 
                                                                                                device = device, peri_flag = peri_flag, proto_flag = proto_flag)
                    peri_test_acc = np.around(peri_test_acc, 4)
                    acc.append(peri_test_acc)

        # *** ***

        acc = np.around(np.mean(acc), 4)
        print(datasets, acc)
        id_dict[datasets] = acc

    return id_dict


# Cross-Modal Identification (Main)
def kfold_cm_id(model, root_pth=config.evaluation['identification'], face_model = None, peri_model = None, device = 'cuda:0', proto_flag = False):
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
            _, inter_face_gal_acc_ethnic = crossmodal_id(model, ethnic_face_gal_load, ethnic_peri_pr_load, device = device, face_model = face_model, peri_model = peri_model, gallery = 'face', proto_flag = proto_flag)
            inter_face_gal_acc_ethnic = np.around(inter_face_gal_acc_ethnic, 4)
            acc_face_gal.append(inter_face_gal_acc_ethnic)

            ethnic_peri_gal_load, ethnic_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', 'periocular', aug='False')
            ethnic_face_pr_load, ethnic_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', 'face', aug='False')
            _, inter_peri_gal_acc_ethnic = crossmodal_id(model, ethnic_face_pr_load, ethnic_peri_gal_load, device = device, face_model = face_model, peri_model = peri_model, gallery = 'peri', proto_flag = proto_flag)
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
                _, cm_face_gal_acc = crossmodal_id(model, face_gal_load, peri_probe_load, device = device, proto_flag = False, face_model = face_model, peri_model = peri_model, gallery = 'face')
                cm_face_gal_acc = np.around(cm_face_gal_acc, 4)
                acc_face_gal.append(cm_face_gal_acc)

                peri_gal_load, peri_dataset = data_loader.gen_data((gallery_path + modal_root[0]), 'test', 'periocular', aug='False')
                face_probe_load, face_dataset = data_loader.gen_data((probes + modal_root[1]), 'test', 'face', aug='False')
                _, cm_peri_gal_acc = crossmodal_id(model, face_probe_load, peri_gal_load, device = device, proto_flag = False, face_model = face_model, peri_model = peri_model, gallery = 'peri')
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
def kfold_mm_id(model, root_pth='./data/', face_model = None, peri_model = None, mode = 'concat', proto_flag = False):
    for datasets in dset_list:
        root_drt = root_pth + datasets + '/**'
        modal_root = ['/peri/', '/face/']
        path_lst = []
        data_loaders = []
        acc = []                

        # *** ***

        if datasets == 'ytf':
            ytf_face_gal_data_load, ytf_face_gal_data_set = data_loader.gen_data((root_pth + 'ytf/gallery/face/'), 'test', 'face', aug='False')
            ytf_face_pr_data_load, ytf_face_pr_data_set = data_loader.gen_data((root_pth + 'ytf/probe/face/'), 'test', 'face', aug='False')
            ytf_peri_gal_data_load, ytf_peri_gal_data_set = data_loader.gen_data((root_pth + 'ytf/gallery/peri/'), 'test', 'periocular', aug='False')            
            ytf_peri_pr_data_load, ytf_peri_pr_data_set = data_loader.gen_data((root_pth + 'ytf/probe/peri/'), 'test', 'periocular', aug='False')        
            _, acc = multimodal_id(model, ytf_face_gal_data_load, ytf_peri_gal_data_load, ytf_face_pr_data_load, ytf_peri_pr_data_load, 
                                    device = device, proto_flag = True, face_model = face_model, peri_model = peri_model, mode = mode)
        elif datasets == 'ethnic':
            ethnic_face_gal_data_load, ethnic_face_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/face/'), 'test', 'face', aug='False')
            ethnic_face_pr_data_load, ethnic_face_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/face/'), 'test', 'face', aug='False')
            ethnic_peri_gal_data_load, ethnic_peri_gal_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/gallery/peri/'), 'test', 'periocular', aug='False')            
            ethnic_peri_pr_data_load, ethnic_peri_pr_data_set = data_loader.gen_data((root_pth + 'ethnic/Recognition/probe/peri/'), 'test', 'periocular', aug='False')        
            _, acc = multimodal_id(model, ethnic_face_gal_data_load, ethnic_peri_gal_data_load, ethnic_face_pr_data_load, ethnic_peri_pr_data_load, 
                                    device = device, proto_flag = False, face_model = face_model, peri_model = peri_model, mode = mode)
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
                    
                    mm_acc1, mm_test_acc = multimodal_id(model, face_data_load_gal, peri_data_load_gal, face_data_load_pr, peri_data_load_pr, 
                                                            device = device, proto_flag = False, face_model = face_model, peri_model = peri_model, mode = mode)
                    mm_test_acc = np.around(mm_test_acc, 4)
                    acc.append(mm_test_acc)
                #     print(i, mm_test_acc)
                # print("Fold:", fold)

        # *** ***

        acc = np.around(np.mean(np.array(acc)), 4)
        print(datasets, acc)
        mm_id_dict[datasets] = acc

    return mm_id_dict


# Identification Function
def validate_identification(model, loader_gallery, loader_test, device = 'cuda:0', peri_flag = False, proto_flag = False):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False
        
    # ***** *****
    
    # Extract gallery features w.r.t. pre-learned model
    gallery_fea = torch.tensor([])
    gallery_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_gallery):

            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

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
    test_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(loader_test):

            x = x.to(device)
            x = model(x, peri_flag = peri_flag)

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
        gal_lbl_proto = torch.tensor([], dtype = torch.int64)
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
    
    gallery_dist = pairwise.cosine_similarity(gallery_fea)
    gallery_pred = np.argmax(gallery_dist, 0)
    gallery_pred = gallery_label[gallery_pred] 
    gallery_acc = sum(gallery_label == gallery_pred) / gallery_label.shape[0]
    
    test_dist = pairwise.cosine_similarity(gallery_fea, test_fea)
    test_pred = np.argmax(test_dist, 0)
    test_pred = gallery_label[test_pred]
    test_acc = sum(test_label == test_pred) / test_label.shape[0]

    # torch.cuda.empty_cache()
    # time.sleep(0.0001)
    
    del model
    time.sleep(0.0001)
    
    return gallery_acc, test_acc


# Cross-Modal Identification Function
def crossmodal_id(model, face_loader, peri_loader, device = 'cuda:0', face_model = None, peri_model = None, gallery = 'face', proto_flag = False):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False

    # ***** *****

    # Extract face features w.r.t. pre-learned model
    face_fea = torch.tensor([])
    face_label = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag = False)
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
    peri_label = torch.tensor([], dtype = torch.int64)
    
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

    # for prototyping
    if proto_flag is True:
        print("WITH Prototyping on periocular and face galleries.")
        # periocular
        peri_lbl_proto = torch.tensor([], dtype = torch.int64)
        peri_fea_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(peri_label):
            # append unique labels to tensor list
            peri_lbl_proto = torch.cat((peri_lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            peri_indices = np.where(peri_label == i)
            peri_feats = torch.tensor([])

            # from index list, append features into temporary peri_feats list
            for j in peri_indices:
                peri_feats = torch.cat((peri_feats, peri_fea[j].detach().cpu()), 0)
            peri_proto_mean = torch.unsqueeze(torch.mean(peri_feats, 0), 0)
            peri_proto_mean = F.normalize(peri_proto_mean, p=2, dim=1)
            peri_fea_proto = torch.cat((peri_fea_proto, peri_proto_mean.detach().cpu()), 0)

        # finally, set periocular feature and label to prototyped ones
        peri_fea, peri_label = peri_fea_proto, peri_lbl_proto

        #### **** ####

        # face
        face_lbl_proto = torch.tensor([], dtype = torch.int64)
        face_fea_proto = torch.tensor([])

        # get unique labels
        for i in torch.unique(face_label):
            # append unique labels to tensor list
            face_lbl_proto = torch.cat((face_lbl_proto, torch.tensor([i], dtype=torch.int64)))

            # get index list where unique labels occur
            face_indices = np.where(face_label == i)
            face_feats = torch.tensor([])

            # from index list, append features into temporary face_feats list
            for j in face_indices:
                face_feats = torch.cat((face_feats, face_fea[j].detach().cpu()), 0)
            face_proto_mean = torch.unsqueeze(torch.mean(face_feats, 0), 0)
            face_proto_mean = F.normalize(face_proto_mean, p=2, dim=1)
            face_fea_proto = torch.cat((face_fea_proto, face_proto_mean.detach().cpu()), 0)

        # finally, set face feature and label to prototyped ones
        face_fea, face_label = face_fea_proto, face_lbl_proto

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
    
    gal_dist = pairwise.cosine_similarity(gal_fea)
    gal_pred = np.argmax(gal_dist, 0)
    gal_pred = gal_label[gal_pred] 
    gal_acc = sum(gal_label == gal_pred) / gal_label.shape[0]
    
    probe_dist = pairwise.cosine_similarity(gal_fea, probe_fea)
    probe_pred = np.argmax(probe_dist, 0)
    probe_pred = gal_label[probe_pred]
    probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]
    
    del model
    time.sleep(0.0001)
    
    return gal_acc, probe_acc


# Multimodal Identification
def multimodal_id(model, face_loader_gal, peri_loader_gal, face_loader_probe, peri_loader_probe, device = 'cuda:0', proto_flag = False, face_model = None, peri_model = None, mode = 'concat'):
    
    # ***** *****
    
    model = model.eval().to(device)
    # model.classify = False

    # ***** *****

    gal_fea = torch.tensor([])
    gal_label = torch.tensor([], dtype = torch.int64)
    probe_fea = torch.tensor([])
    probe_label = torch.tensor([], dtype = torch.int64)

    # ***** *****

    # GALLERY
    # Extract face features w.r.t. pre-learned model
    face_fea_gal = torch.tensor([])
    face_label_gal = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader_gal):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag = False)
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
    peri_label_gal = torch.tensor([], dtype = torch.int64)
    
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
    face_label_probe = torch.tensor([], dtype = torch.int64)
    
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(face_loader_probe):
            
            x = x.to(device)
            if not face_model is None:
                face_model = face_model.eval().to(device)
                x = face_model(x, peri_flag = False)
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
    peri_label_probe = torch.tensor([], dtype = torch.int64)
    
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
        peri_lbl_proto = torch.tensor([], dtype = torch.int64)
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
        face_lbl_proto = torch.tensor([], dtype = torch.int64)
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
    
    gal_dist = pairwise.cosine_similarity(gal_fea)
    gal_pred = np.argmax(gal_dist, 0)
    gal_pred = gal_label[gal_pred] 
    # gal_acc = sum(gal_label == gal_pred) / gal_label.shape[0]

    probe_dist = pairwise.cosine_similarity(gal_fea, probe_fea)
    probe_pred = np.argmax(probe_dist, 0)
    probe_pred = gal_label[probe_pred]
    probe_acc = sum(probe_label == probe_pred) / probe_label.shape[0]

    torch.cuda.empty_cache()
    # time.sleep(0.0001)
    
    del model
    time.sleep(0.0001)
    
    return 1, probe_acc


if __name__ == '__main__':
    embd_dim = 512
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mm_mode = 'concat'

    load_model_path = './models/CB_Net/best_model/CB_Net.pth'
    model = net.CB_Net(embedding_size = embd_dim, do_prob=0.0).eval().to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)

    peri_id_dict = kfold_identification(model, root_pth=config.evaluation['identification'], modal = 'periocular', peri_flag = True, device = device)
    peri_id_dict = get_avg(peri_id_dict)
    peri_id_dict = copy.deepcopy(peri_id_dict)
    print('Average (Periocular):', peri_id_dict['avg'])
    print('Periocular:', peri_id_dict)    

    face_id_dict = kfold_identification(model, root_pth = config.evaluation['identification'], modal = 'face', peri_flag = False, device = device)
    face_id_dict = get_avg(face_id_dict)
    face_id_dict = copy.deepcopy(face_id_dict)
    print('Average (Face):', face_id_dict['avg'])
    print('Face:', face_id_dict)    

    cm_id_dict_f, cm_id_dict_p = kfold_cm_id(model, root_pth = config.evaluation['identification'], face_model = None, peri_model = None, device = device)
    cm_id_dict_p, cm_id_dict_f = get_avg(cm_id_dict_p), get_avg(cm_id_dict_f)
    cm_id_dict_p = copy.deepcopy(cm_id_dict_p)
    cm_id_dict_f = copy.deepcopy(cm_id_dict_f)
    print('Average (Periocular-Face):', cm_id_dict_p['avg'], cm_id_dict_f['avg'])
    print('Cross-Modal (Periocular Gallery, Face Gallery):', cm_id_dict_p, cm_id_dict_f)    

    mm_id_dict = kfold_mm_id(model, root_pth='./data/', face_model = None, peri_model = None, mode = 'concat')
    mm_id_dict = get_avg(cm_id_dict_f)
    print('Average (Periocular+Face):', mm_id_dict['avg'])
    print('Multimodal (Face):', mm_id_dict)    