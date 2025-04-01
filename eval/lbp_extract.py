from skimage import feature
import numpy as np
import cv2
import glob, os, sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.metrics import pairwise
from torch.nn import functional as F
from configs import config


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius)
        return lbp


def preprocess_img(params, imagePath):
    img = cv2.imread(imagePath, 0)
    img = cv2.resize(img, (112, 112))
    a, b = params[0], params[1]
    desc = LocalBinaryPatterns(a, b)

    lbp = desc.describe(img)
    return lbp


def read_images(params, type_root, root=None):
    img_path = '**/*.jpg'
    features = torch.Tensor([])
    label_nms = []

    if not root is None:
        dir_root = root
    else:
        dir_root = './data/pearson/' + type_root    
    
    for name in tqdm(glob.glob(os.path.join(dir_root, img_path))):
        feats = preprocess_img(params, name)
        feats = torch.unsqueeze(torch.Tensor(feats), dim=0)
        features = torch.cat((features, feats), 0)
        label_nm = name.split('/')[-2]
        label_nms.append(label_nm)

    # convert label text to integers
    le = preprocessing.LabelEncoder()
    labels = torch.Tensor(le.fit_transform(label_nms))

    return features, labels


def identification(loader_gallery, loader_test):    
    gallery_fea = loader_gallery[0]
    gallery_label = loader_gallery[1]  
    test_fea = loader_test[0]
    test_label = loader_test[1]
    
    gallery_fea = F.normalize(gallery_fea, p=2, dim=1)
    test_fea = F.normalize(test_fea, p=2, dim=1)

    # Calculate gallery_acc and test_acc
    gallery_label = np.reshape(np.array(gallery_label), -1)
    test_label = np.reshape(np.array(test_label), -1)
    
    test_dist = pairwise.cosine_similarity(gallery_fea, test_fea)
    test_pred = np.argmax(test_dist, 0)
    test_pred = gallery_label[test_pred]
    test_acc = sum(test_label == test_pred) / test_label.shape[0]
    
    return test_acc


if __name__ == "__main__":  
    params = (8, 1)

    # read images and extract the features
    face_gallery_emb, face_gallery_lbl = read_images(params, 'face', config.ethnic['face_gallery'])
    face_probe_emb, face_probe_lbl = read_images(params, 'face', config.ethnic['face_probe'])
    peri_gallery_emb, peri_gallery_lbl = read_images(params, 'peri', config.ethnic['peri_gallery'])
    peri_probe_emb, peri_probe_lbl = read_images(params, 'peri', config.ethnic['peri_probe'])

    # perform identification
    face_gal_face_probe = identification((torch.flatten(face_gallery_emb, start_dim=1), face_gallery_lbl), (torch.flatten(face_probe_emb, start_dim=1), face_probe_lbl))
    print('Intra-Modal Face:', face_gal_face_probe)     

    peri_gal_peri_probe = identification((torch.flatten(peri_gallery_emb, start_dim=1), peri_gallery_lbl), (torch.flatten(peri_probe_emb, start_dim=1), peri_probe_lbl))
    print('Intra-Modal Periocular:', peri_gal_peri_probe)

    face_gal_peri_probe = identification((torch.flatten(face_gallery_emb, start_dim=1), face_gallery_lbl), (torch.flatten(peri_probe_emb, start_dim=1), peri_probe_lbl))
    print('Cross-Modal (Face Gallery):', face_gal_peri_probe)  

    peri_gal_face_probe = identification((torch.flatten(peri_gallery_emb, start_dim=1), peri_gallery_lbl), (torch.flatten(face_probe_emb, start_dim=1), face_probe_lbl))
    print('Cross-Modal (Periocular Gallery)):', peri_gal_face_probe)