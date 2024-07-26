# import the necessary packages
from skimage import feature
import numpy as np
import cv2
import glob, os, sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.metrics import pairwise
import time
from torch.nn import functional as F
from configs import config
import matplotlib as plt


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
  

def describe_images(imagePath, params):
  a, b = params[0], params[1]
  desc = LocalBinaryPatterns(a, b)
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hist = desc.describe(gray)

  return hist


def preprocess_img(params, imagePath):
  img = cv2.imread(imagePath, 0)
  img = cv2.resize(img, (112, 112))
  a, b = params[0], params[1]
  desc = LocalBinaryPatterns(a, b)
  # rects = face_detection(img)
  # print(rects)
  # for (x , y , w , h) in rects:
  #   face = img[y:y+h , x:x+w]
    # print(face.shape)

  # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
  # plt.imshow(face , cmap="gray")
  # print(face.shape)
  # face = np.array(face)
  
  lbp = desc.describe(img)
  return lbp # , hist


def read_images(params, type_root, root = None):
  features = torch.Tensor([])
  label_nms = []

  if not root is None:
    dir_root = root
  else:
    dir_root = './data/test1_pearson/' + type_root
  img_path = '**/*.jpg'
  for name in tqdm(glob.glob(os.path.join(dir_root, img_path))):
    feats = preprocess_img(params, name)
    feats = torch.unsqueeze(torch.Tensor(feats), dim=0)

    features = torch.cat((features, feats), 0)
    label_nm = name.split('/')[-2]
    label_nms.append(label_nm)
    # print(name)
  
  # convert label text to integers
  le = preprocessing.LabelEncoder()
  labels = torch.Tensor(le.fit_transform(label_nms))

  return features, labels


# identification
def validate_identification(loader_gallery, loader_test):    
    gallery_fea = loader_gallery[0]
    gallery_label = loader_gallery[1]
    
    time.sleep(0.0001)    
  
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
  desc = LocalBinaryPatterns(8 , 2)
  face_gallery_emb, face_gallery_lbl = read_images(params, 'face', config.ethnic['face_gallery'])
  face_probe_emb, face_probe_lbl = read_images(params, 'face', config.ethnic['face_probe'])

  torch.save(face_gallery_emb, './data/lbp/face_gallery_emb.pt')
  torch.save(face_gallery_lbl, './data/lbp/face_gallery_lbl.pt')
  torch.save(face_probe_emb, './data/lbp/face_probe_emb.pt')
  torch.save(face_probe_lbl, './data/lbp/face_probe_lbl.pt')

  face_gal = validate_identification((torch.flatten(face_gallery_emb, start_dim = 1), face_gallery_lbl), (torch.flatten(face_probe_emb, start_dim = 1), face_probe_lbl))
  print(face_gal)  

  peri_gallery_emb, peri_gallery_lbl = read_images(params, 'peri', config.ethnic['peri_gallery'])
  peri_probe_emb, peri_probe_lbl = read_images(params, 'peri', config.ethnic['peri_probe'])  

  torch.save(peri_gallery_emb, './data/lbp/peri_gallery_emb.pt')
  torch.save(peri_gallery_lbl, './data/lbp/peri_gallery_lbl.pt')
  torch.save(peri_probe_emb, './data/lbp/peri_probe_emb.pt')
  torch.save(peri_probe_lbl, './data/lbp/peri_probe_lbl.pt')
  
  probe_gal = validate_identification((torch.flatten(peri_gallery_emb, start_dim = 1), peri_gallery_lbl), (torch.flatten(peri_probe_emb, start_dim = 1), peri_probe_lbl))
  print(probe_gal)

  face_gal = validate_identification((torch.flatten(face_gallery_emb, start_dim = 1), face_gallery_lbl), (torch.flatten(peri_probe_emb, start_dim = 1), peri_probe_lbl))
  print(face_gal)  

  probe_gal = validate_identification((torch.flatten(peri_gallery_emb, start_dim = 1), peri_gallery_lbl), (torch.flatten(face_probe_emb, start_dim = 1), face_probe_lbl))
  print(probe_gal)