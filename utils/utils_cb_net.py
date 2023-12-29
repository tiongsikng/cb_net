
# Source : https://github.com/adambielski/siamese-triplet/

import numpy as np
# import random
# import sys

from PIL import Image
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import random
import configs.params as params

device = params.device

# **** *** *** ***

class BalancedBatchSampler(BatchSampler):
    
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = torch.tensor(labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # print(self.label_to_indices)
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):        
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # seed is used to enable same batches for both streams
            np.random.seed(self.count)
            random.seed(self.count)
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            # classes = np.random.choice(self.labels_set, self.n_classes, replace = True)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
# **** *** *** ***

class BalancedBatchSampler_OPT(BatchSampler):
    
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = torch.tensor(labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # print(self.label_to_indices)
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            classes = np.random.choice(self.labels_set, self.n_classes, replace = True)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
# **** *** *** ***

class OnlineContrastiveLoss(nn.Module):
    
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.to(device)
            negative_pairs = negative_pairs.to(device)
        
        positive_loss = 1 - torch.diag(torch.mm(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]].t()))
        negative_loss = 1 - torch.diag(torch.mm(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]].t()))
        
        ap = positive_loss.mean().cpu()
        an = negative_loss.mean().cpu()
        
        # ***
        
        negative_loss = F.relu(self.margin - negative_loss)

        loss = torch.cat( [ positive_loss, 
                            negative_loss ], dim=0)
        
        # ***
        
        return loss.mean(), ap, an
    

# **** *** *** ***

class OnlineTripletLoss(nn.Module):
    
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin = 0.7, ap_weight = 0.0, triplet_selector = None):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.ap_weight = ap_weight
        self.triplet_selector = triplet_selector
        
    def forward(self, embeddings, target, target_min = 0, k = 1, ap_flag = False):

        # if k == 0:
        #    k = 1
          
        # if ep in ([ 12, 24, 36, 48 ]):
        #    self.ap_weight *= 0.9
        
        triplets = self.triplet_selector.get_triplets(embeddings, target, target_min, k)
        
        # torch.set_printoptions(profile="full")
        # print(triplets.size())
        # print(triplets)
        
        if embeddings.is_cuda:
            triplets = triplets.to(device)
            
        '''
        ap_distances = 1 - torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]], 
                                                           embeddings[triplets[:, 1]].t())), min=-1., max=1.)

        an_distances = 1 - torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]], 
                                                           embeddings[triplets[:, 2]].t())), min=-1., max=1.)
        '''
        
        ap_distances = torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]], 
                                                       embeddings[triplets[:, 1]].t())), min=-1., max=1.)

        an_distances = torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]], 
                                                       embeddings[triplets[:, 2]].t())), min=-1., max=1.)
        
        
        
        
        # ***        
        # This is used for direct AP penalization, remove the statement after the ap_flag is True to use this.
        # ap_distances = ( ap_distances / (self.margin) )

        # ***
        # This is used to linked the negative directly to the anchor (extended AN), which does not rely on relationship between P and N. To enable, also
        # modify the codes in FunctionNegativeTripletSelector.

        # an_distances_ext = 1 - torch.clamp(torch.diag(torch.mm(embeddings[triplets[:, 0]],
        #                                                        embeddings[triplets[:, -1]].t())), min=-1., max=1.)

        # an_distances_ext[ triplets[:, 2 ]  == triplets[:, -1] ] = 1e-10
                
        # losses = F.relu( ap_distances - an_distances + self.margin ) + F.relu( 1.0 - an_distances_ext )

        # ***

        # losses = F.relu( ap_distances - an_distances + self.margin ) 
        losses = torch.exp( ( an_distances - ap_distances ) / self.margin )
        
        # for stability, this statement is removed.
        # losses = losses[ losses > 0 ]
        
        ap = ( 1.0 - ap_distances ).mean().cpu()
        an = ( 1.0 - an_distances ).mean().cpu()
        
        # *** 
                        
        # Introduce anchor-positive supression term
        losses_reg = 0.0

        if ap_flag is True:
            
            # # losses_reg = torch.mean( torch.exp( ap_distances / self.margin ) - an_distances ) * self.ap_weight
            # losses_reg = torch.sum( ( ap_distances / self.margin ) - an_distances ) * self.ap_weight
            # # losses_reg = torch.sum( ( ap_distances / self.margin + 0.001 ) - an_distances ) * self.ap_weight
            # losses_reg = F.relu( losses_reg )

            # losses_reg = torch.sum( F.relu( torch.sqrt( ap_distances / self.margin ) - an_distances ) ) * self.ap_weight
            losses_reg = torch.sum( torch.exp( an_distances - ap_distances * self.margin ) ) * self.ap_weight
        
        # ***

        # losses = losses[ losses > 0 ]
        losses = losses.mean() + losses_reg
        
        return losses, ap, an    
        
'''
def pdist(vectors):
    
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    
    return distance_matrix
'''

# **** *** *** ***

class PairSelector:
    
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError

class AllPositivePairSelector(PairSelector):
    
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        
        labels = labels.cpu().data.numpy()
        
        # ***
        
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        
        # ***
        
        if self.balance:
            negative_pairs = negative_pairs[ torch.randperm(len(negative_pairs))[:len(positive_pairs)] ]
            
        # ***

        return positive_pairs, negative_pairs

    
class HardNegativePairSelector(PairSelector):
    
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        
        if self.cpu:
            embeddings = embeddings.cpu()
        
        # distance_matrix = pdist(embeddings)
        # print(distance_matrix)
       
        distance_matrix = 1 - torch.clamp(torch.mm(embeddings, embeddings.t()),min=-1.0, max=1.0)
        # distance_matrix = 1 - torch.mm(embeddings, embeddings.t())

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)

        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        
        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

    
# **** *** *** ***

class TripletSelector:
    
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        
        print('here here here here')
        
        labels = labels.cpu().data.numpy()
        triplets = []
        
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

def hardest_negative(loss_values, k):
    
    '''
    hard_negative = np.argsort(loss_values)[::-1][k]

    return hard_negative if loss_values[hard_negative] > 0 else None
    '''
    hard_negative = np.argsort(loss_values)[::-1][:k]

    # return np.random.choice(hard_negative) if len(hard_negative) > 0 else None
    return hard_negative if len(hard_negative) > 0 else None
    
    '''
    hard_negative = np.argsort(loss_values)[::-1][:k]
    hard_negative = hard_negative[loss_values[hard_negative] > 0] 
    
    return hard_negative if len(hard_negative) > 0 else None
    '''
    
    '''
    # hard_negative = np.argsort(loss_values)[::-1][k]
    # return hard_negative if loss_values[hard_negative] > 0 else None
    # return only the most negative index
    # hard_negative = np.argmax(loss_values)
    # return hard_negative if loss_values[hard_negative] > 0 else None
    # hard_negative = np.argsort(loss_values)[::-1][:k]
    # assert(len(hard_negative)<=k)
    # hard_negative = hard_negative[loss_values[hard_negative] > 0] 
    # if k < len(hard_negative):
    #    hard_negative = hard_negative[:k]
    # print('k :', k)
    # print('hard negative :', hard_negative)
    # return hard_negative
    '''
    
def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

class FunctionNegativeTripletSelector(TripletSelector):
    
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu = True):
        
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    # def get_triplets(self, embeddings, labels):
    def get_triplets(self, embeddings, labels, label_min = 0, k = 1):
        
        if self.cpu:
            embeddings = embeddings.cpu()
        
        distance_matrix = 1 - torch.clamp(torch.mm(embeddings, embeddings.t()), min = -1.0, max = 1.0)
        distance_matrix = distance_matrix.cpu()
        
        labels = labels.cpu().data.numpy()
        triplets = []
        # aps = []

        # set() : no duplication
        for label in set(labels):
            
            if label >= label_min:
            
                label_mask = (labels == label)
                label_indices = np.where(label_mask)[0]
                
                # If no ap pairs formed
                if len(label_indices) < 2:
                    continue
                    
                negative_indices = np.where(np.logical_not(label_mask))[0]
               
                # All anchor-positive pairs
                # anchor_positives = list(combinations(label_indices, 2))  
                # using zip() + list slicing  
                # to perform pair iteration in list  
                # https://www.geeksforgeeks.org/python-pair-iteration-in-list/
                
                # Randomize AP pairs 
                # print(label, int(torch.sum(distance_matrix)))
                np.random.seed(int(torch.sum(distance_matrix))*label)
                label_indices = np.random.permutation(label_indices)
                
                #
                #
                #
                # Use all AP pairs for deeper networks
                # Otherwise use only AP subset
                anchor_positives = list(combinations(label_indices, 2))
                anchor_positives = np.array(anchor_positives)

                # label_indices = list(label_indices)
                # anchor_positives = list(zip(label_indices, label_indices[1:] + label_indices[:1])) 
                # anchor_positives = np.array(anchor_positives)

                '''
                # print('Distance Matrix Size : ', distance_matrix.size())
                # print(type(label_indices))
                # print('AP Indices : ', label_indices)
                # print('AP Pairs : ', anchor_positives)
                '''

                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                    
                    an_distance = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])),\
                                                  torch.LongTensor(negative_indices)] 

                    loss_values = ap_distance - an_distance + self.margin
                    
                    loss_values = loss_values.data.cpu().numpy()
                    
                    hard_negative = self.negative_selection_fn(loss_values,k)

                    # ***

                    # an_distance_ext = distance_matrix[torch.LongTensor(np.array([anchor_positive[1]])),\
                    #                               torch.LongTensor(negative_indices)] 

                    # hard_negative_ext = np.argsort(an_distance_ext.cpu().data.numpy())
                    # hard_negative_ext = np.random.choice(hard_negative_ext)
                    
                    # *** 
                    
                    '''
                    if hard_negative is not None:
                        for hn in hard_negative:
                            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[hn]])
                    '''    
                    
                    # aps.append([anchor_positive[0], anchor_positive[1]])
                    
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        # hard_negative_ext = negative_indices[hard_negative_ext]
                        triplets.append([anchor_positive[0], anchor_positive[1], int(hard_negative)]) #, hard_negative_ext
                    
                # print('AP Indices : ', label_indices)
                # print('AP Pairs : ', anchor_positives)
                # print('Triplets : ', triplets)
                # print('type(Triplets) : ', type(triplets))
                    
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
            
        triplets = np.array(triplets)
        # aps = np.array(aps)
        
        return torch.LongTensor(triplets) # , torch.LongTensor(aps)

def HardestNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative,cpu=cpu)

def RandomNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=random_hard_negative,cpu=cpu)

def SemihardNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=lambda x: semihard_negative(x, margin),cpu=cpu)


# **** *** *** ***

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'
    
class AverageNonzeroTripletsMetric(Metric):
    
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'