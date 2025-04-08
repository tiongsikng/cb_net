# *** *** *** ***
# Boiler Codes - Import Dependencies

if __name__ == '__main__': # used for Windows freeze_support() issues
    from torch.optim import lr_scheduler
    import torch.nn as nn
    import torch.optim as optim
    import torch
    import torch.multiprocessing
    import pickle
    import numpy as np
    import random
    import copy
    import os, sys, glob, shutil
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import json
    import argparse
    from torchsummary import summary

    sys.path.insert(0, os.path.abspath('.'))
    from configs.params import *
    from configs import params
    from configs import datasets_config as config
    from data import data_loader as data_loader
    import network.cb_net as net
    from network.logits import CosFace
    from utils.utils_cb_net import OnlineTripletLoss, HardestNegativeTripletSelector
    import train
    from eval import cmc_eval_identification as identification
    from eval import roc_eval_verification as verification
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Imported.")

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--method', default=params.method, type=str,
                        help='method (backbone)')
    parser.add_argument('--remarks', default=params.remarks, type=str,
                        help='additional remarks')
    parser.add_argument('--write_log', default=params.write_log, type=bool,
                        help='flag to write logs')
    parser.add_argument('--dim', default=params.dim, type=int, metavar='N',
                        help='embedding dimension')
    parser.add_argument('--epochs', default=params.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=params.lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--w_decay', '--w_decay', default=params.w_decay, type=float,
                        metavar='Weight Decay', help='weight decay')
    parser.add_argument('--dropout', '--dropout', default=params.dropout, type=float,
                        metavar='Dropout', help='dropout probability')
    parser.add_argument('--pretrained', default='/home/tiongsik/Python/conditional_biometrics/models/pretrained/MobileFaceNet_AF_S30.0_M0.4_D512_EP16.pth', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    args = parser.parse_args()

    # Determine if an nvidia GPU is available
    device = params.device

    # For reproducibility, Seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    file_main_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    print('Running on device: {}'.format(device))
    start_ = datetime.now()
    start_string = start_.strftime("%Y%m%d_%H%M%S")

    # *** *** *** ***
    # Load Dataset and Display Other Parameters

    # Face images
    face_train_dir = config.trainingdb['face_train']
    face_loader_train, face_train_set = data_loader.gen_data(face_train_dir, 'train_rand', type='face', aug='True')
    face_loader_train_tl, face_train_tl_set = data_loader.gen_data(face_train_dir, 'train', type='face', aug='True')

    # Periocular Images
    peri_train_dir = config.trainingdb['peri_train']
    peri_loader_train, peri_train_set = data_loader.gen_data(peri_train_dir, 'train_rand', type='periocular', aug='True')
    peri_loader_train_tl, peri_train_tl_set = data_loader.gen_data(peri_train_dir, 'train', type='periocular', aug='True')

    # Validation Periocular (Gallery/Val + Probe/Test)
    peri_val_dir = config.trainingdb['peri_val']
    peri_loader_val, peri_val_set = data_loader.gen_data(peri_val_dir, 'test', type='periocular', aug='False')
    peri_test_dir = config.trainingdb['peri_test']
    peri_loader_test, peri_test_set = data_loader.gen_data(peri_test_dir, 'test', type='periocular', aug='False')

    # Validation Face (Gallery/Val + Probe/Test)
    face_val_dir = config.trainingdb['face_val']
    face_loader_val, face_val_set = data_loader.gen_data(face_val_dir, 'test', type='face', aug='False')
    face_test_dir = config.trainingdb['face_test']
    face_loader_test, face_test_set = data_loader.gen_data(face_test_dir, 'test', type='face', aug='False')

    # Test Periocular (Ethnic)
    ethnic_peri_gallery_dir = config.ethnic['peri_gallery']
    ethnic_peri_probe_dir = config.ethnic['peri_probe']
    ethnic_peri_val_loader, ethnic_peri_val_set = data_loader.gen_data(ethnic_peri_gallery_dir, 'test', type='periocular')
    ethnic_peri_test_loader, ethnic_peri_test_set = data_loader.gen_data(ethnic_peri_probe_dir, 'test', type='periocular')

    # Test Face (Ethnic)
    ethnic_face_gallery_dir = config.ethnic['face_gallery']
    ethnic_face_probe_dir = config.ethnic['face_probe']
    ethnic_face_val_loader, ethnic_face_val_set = data_loader.gen_data(ethnic_face_gallery_dir, 'test', type='face')
    ethnic_face_test_loader, ethnic_face_test_set = data_loader.gen_data(ethnic_face_probe_dir, 'test', type='face')

    # Set and Display all relevant parameters
    print('\n***** Face ( Train ) *****\n')
    face_num_train = len(face_train_set)
    face_num_sub = len(face_train_set.classes)
    print(face_train_set)
    print('Num. of Sub.\t\t:', face_num_sub)
    print('Num. of Train. Imgs (Face) \t:', face_num_train)

    print('\n***** Periocular ( Train ) *****\n')
    peri_num_train = len(peri_train_set)
    peri_num_sub = len(peri_train_set.classes)
    print(peri_train_set)
    print('Num. of Sub.\t\t:', peri_num_sub)
    print('Num. of Train Imgs (Periocular) \t:', peri_num_train)

    print('\n***** Periocular ( Validation (Gallery) ) *****\n')
    peri_num_val = len(peri_val_set)
    print(peri_val_set)
    print('Num. of Sub.\t\t:', len(peri_val_set.classes))
    print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    print('\n***** Periocular ( Validation (Probe) ) *****\n')
    peri_num_test = len(peri_test_set)
    print(peri_test_set)
    print('Num. of Sub.\t\t:', len(peri_test_set.classes))
    print('Num. of Test Imgs (Periocular) \t:', peri_num_test)

    # print('\n***** Face ( Validation (Gallery) ) *****\n')
    # peri_num_val = len(face_val_set)
    # print(face_val_set)
    # print('Num. of Sub.\t\t:', len(face_val_set.classes))
    # print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    # print('\n***** Face ( Test (Probe) ) *****\n')
    # face_num_test = len(face_test_set)
    # print(face_test_set)
    # print('Num. of Sub.\t\t:', len(face_test_set.classes))
    # print('Num. of Test Imgs (Periocular) \t:', face_num_test)

    print('\n***** Other Parameters *****\n')
    print('Start Time \t\t: ', start_string)
    print('Method (Backbone)\t: ', args.method)
    print('Remarks\t\t\t: ', args.remarks)
    print('Net. Descr.\t\t: ', net_descr)
    print('Seed\t\t\t: ', seed)
    print('Batch # Sub.\t\t: ', batch_sub)
    print('Batch # Samp.\t\t: ', batch_samp)
    print('Batch Size\t\t: ', batch_size)
    print('Emb. Dimension\t\t: ', args.dim)
    print('# Epoch\t\t\t: ', epochs)
    print('Learning Rate\t\t: ', args.lr)
    print('LR Scheduler\t\t: ', lr_sch)
    print('Weight Decay\t\t: ', args.w_decay)
    print('Dropout Prob.\t\t: ', args.dropout)
    print('BN Flag\t\t\t: ', bn_flag)
    print('BN Momentum\t\t: ', bn_moment)
    print('Scaling\t\t\t: ', cf_s)
    print('Margin\t\t\t: ', cf_m)
    print('Trip. Loss ID\t\t: ', tl_id)
    print('Trip. Loss Regularizer\t: ', tl_ap)
    print('Trip. Loss Margin\t: ', tl_m)
    print('Trip. Loss #k\t\t: ', tl_k)
    print('Trip. Loss Alpha\t: ', tl_alpha)
    print('Save Flag\t\t: ', save)
    print('Log Writing\t\t: ', args.write_log)

    # *** *** *** ***
    # Load Pre-trained Model, Define Loss and Other Hyperparameters for Training

    print('\n***** *****\n')
    print('Loading Pretrained Model' )  
    print()

    train_mode = 'eval'
    model = net.CB_Net(embedding_size=args.dim, do_prob=args.dropout).eval().to(device)

    load_model_path = args.pretrained
    state_dict_loaded = model.state_dict()
    state_dict_pretrained = torch.load(load_model_path, map_location=device)
    # state_dict_pretrained = torch.load(load_model_path, map_location=device)['state_dict']
    state_dict_temp = {}

    for k in state_dict_loaded:
        if 'conv_6_gap' not in k: # Adaptive layer for periocular
            state_dict_temp[k] = state_dict_pretrained[k]
            # state_dict_temp[k] = state_dict_pretrained['backbone.'+k]
        else:
            print(k, 'not loaded!')
    state_dict_loaded.update(state_dict_temp)
    model.load_state_dict(state_dict_loaded)
    del state_dict_loaded, state_dict_pretrained, state_dict_temp

    # for multiple GPU usage, set device in params to torch.device('cuda') without specifying GPU ID.
    # model = torch.nn.DataParallel(model).cuda()
    ####

    # in_features = 7 * 7 * 512 # model.linear.in_features
    # out_features = 1024       # model.linear.out_features * 2
    in_features  = model.linear.in_features
    out_features = args.dim 

    # for MobileFaceNet
    model.linear = nn.Linear(in_features, out_features, bias=True)                      # Deep Embedding Layer
    model.bn = nn.BatchNorm1d(out_features, eps=1e-5, momentum=0.1, affine=True) # BatchNorm1d Layer

    #### model summary
    # torch.cuda.empty_cache()
    # import os
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # summary(model.to(device),(3,112,112))

    # *** ***

    # print('\n***** *****\n')
    print('Appending Face-FC to model ( w.r.t. Face ) ... ' )  
    face_fc = CosFace(in_features=out_features, out_features=face_num_sub, s=cf_s, m=cf_m).eval().to(device)

    # *** ****

    # print('\n***** *****\n')
    print('Appending Peri-FC to model ( w.r.t. Periocular ) ... ' )
    peri_fc = CosFace(in_features=out_features, out_features=peri_num_sub, s=cf_s, m=cf_m).eval().to(device)

    # **********

    print('Re-Configuring Model, Face-FC, and Peri-FC ... ' ) 
    print()

    # *** ***

    # model : Determine parameters to be freezed, or unfreezed
    for name, param in model.named_parameters():
        param.requires_grad = True        

    # model : Display all learnable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('model (requires grad)\t:', name)

    # print('model\t: ALL Parameters')
            
    # *** 

    # model : Freeze or unfreeze BN parameters
    for name, layer in model.named_modules():
        if isinstance(layer,torch.nn.BatchNorm2d):
            #or isinstance(layer,torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.LayerNorm):
            # ***
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True
            # *** 

    if bn_flag == -1:
        print('model\t: EXCLUDE BatchNorm2D Parameters')
    elif bn_flag == 0 or bn_flag == 1:
        print('model\t: INCLUDE BatchNorm2D.weight & bias')

    # *** ***

    # face_fc : Determine parameters to be freezed, or unfreezed
    for param in face_fc.parameters():
        if face_fc_ce_flag is True:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # face_fc : Display all learnable parameters
    print()
    print('face_fc\t:', face_fc)
    for name, param in face_fc.named_parameters():
        if param.requires_grad:   
            print('face_fc\t:', name)

    # *** ***

    # peri_fc : Determine parameters to be freezed, or unfreezed
    for param in peri_fc.parameters():
        if peri_fc_ce_flag is True:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # peri_fc : Display all learnable parameters
    print('peri_fc\t:', peri_fc)
    for name, param in peri_fc.named_parameters():
        if param.requires_grad:
            print('peri_fc\t:', name)

    # ********** 
    # Set an optimizer, scheduler, etc.
    loss_fn = { 'loss_ce' : torch.nn.CrossEntropyLoss(),
                'loss_tl' : OnlineTripletLoss(tl_m, tl_ap, HardestNegativeTripletSelector(tl_m))}
            
    parameters_backbone = [p for p in model.parameters() if p.requires_grad]
    parameters_face_fc = [p for p in face_fc.parameters() if p.requires_grad]
    parameters_peri_fc = [p for p in peri_fc.parameters() if p.requires_grad]

    optimizer = optim.AdamW([   {'params': parameters_backbone},
                                {'params': parameters_face_fc, 'lr': lr*10, 'weight_decay': args.w_decay},
                                {'params': parameters_peri_fc, 'lr': lr*10, 'weight_decay': args.w_decay},
                            ], lr = args.lr, weight_decay = args.w_decay)

    # opt_params = list(model.parameters()) + list(face_fc.parameters()) + list(peri_fc.parameters())
    # optimizer = optim.AdamW(opt_params, lr=args.lr, weight_decay=args.w_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_sch, gamma=0.1)

    metrics = { 'fps': train.BatchTimer(), 'acc': train.accuracy}

    net_params = { 'network' : net_descr, 'method' : args.method, 'remarks' : args.remarks,
                'face_fc_ce_flag' : face_fc_ce_flag, 'peri_fc_ce_flag' : peri_fc_ce_flag, 'face_peri_loss_flag' : face_peri_loss_flag,
                'face_num_sub' : face_num_sub, 'peri_num_sub': peri_num_sub, 'scale' : cf_s, 'margin' : cf_m,
                'lr' : args.lr, 'lr_sch': lr_sch, 'w_decay' : args.w_decay, 'dropout' : args.dropout,
                'tl_k' : tl_k, 'tl_ap' : tl_ap, 'tl_id' : tl_id, 'tl_alpha' : tl_alpha, 
                'batch_sub' : batch_sub, 'batch_samp' : batch_samp, 'batch_size' : batch_size, 'dims' : args.dim, 'seed' : seed }

    # *** *** *** ***
    #### Model Training

    #### Define Logging
    train_mode = 'train'
    log_folder = "./logs/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks)
    if not os.path.exists(log_folder) and args.write_log is True:
        os.makedirs(log_folder)
    log_nm = log_folder + "/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks) + ".txt"

    # Files Backup
    if args.write_log is True: # only backup if there is log
        # copy main and training files as backup
        for files in glob.glob(os.path.join(file_main_path, '*')):
            if '__' not in files: # ignore __pycache__
                shutil.copy(files, log_folder)
                print(files)
        # networks and logits
        py_extension = '.py'
        desc = file_main_path.split('/')[-1]
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'configs'), 'params' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'cb_net' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'logits' + py_extension), log_folder)

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write(str(net_descr) + "\n")
        file.write('Training started at ' + str(start_) + ".\n\n")
        file.write('Model parameters: \n')
        file.write(json.dumps(net_params) + "\n\n")
        file.close()

    # *** ***
    #### Begin Training

    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0

    peri_best_test_acc = 0
    peri_best_pr_test_acc = 0

    best_model = copy.deepcopy(model.state_dict())
    best_face_fc = copy.deepcopy(face_fc.state_dict())
    best_peri_fc = copy.deepcopy(peri_fc.state_dict())

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    model.eval().to(device)
    face_fc.eval().to(device)
    peri_fc.eval().to(device)

    #### Test before training

    # _, ethnic_peri_test_acc = identification.validate_identification(model, ethnic_peri_val_loader, ethnic_peri_test_loader, device=device, peri_flag=True)
    # ethnic_peri_test_acc = np.around(ethnic_peri_test_acc, 4)
    # print('Test Rank-1 IR (Ethnic - Periocular)\t: ', ethnic_peri_test_acc)   

    # _, ethnic_face_test_acc = identification.validate_identification(model, ethnic_face_val_loader, ethnic_face_test_loader, device=device, peri_flag=False)
    # ethnic_face_test_acc = np.around(ethnic_face_test_acc, 4)
    # print('Test Rank-1 IR (Ethnic - Face)\t: ', ethnic_face_test_acc)   

    #### Begin Training
    for epoch in range(epochs):    
        print()
        print()        
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train().to(device)
        face_fc.eval().to(device)
        peri_fc.eval().to(device)
        
        if face_fc_ce_flag is True:
            face_fc.train().to(device)    
        if peri_fc_ce_flag is True:
            peri_fc.train().to(device)
        
        # Use running_stats for training and testing
        if bn_flag != 1:
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):         
                    layer.eval()
                
        train_acc, loss = train.run_train(model, face_fc = face_fc, peri_fc = peri_fc, 
                                            face_loader = face_loader_train, peri_loader = peri_loader_train, face_loader_tl = face_loader_train_tl, peri_loader_tl = peri_loader_train_tl,
                                            net_params = net_params, loss_fn = loss_fn, optimizer = optimizer, 
                                            scheduler = scheduler, batch_metrics = metrics, 
                                            show_running = True, device = device, writer = writer)   
        print('Loss : ', loss)
        # *** ***    
        model.eval().to(device)
        face_fc.eval().to(device)
        peri_fc.eval().to(device)    
        # *****
        
        # print('Periocular')
        peri_val_acc = identification.intramodal_id(model, peri_loader_val, peri_loader_test, device=device, peri_flag=True)
        peri_val_acc = np.around(peri_val_acc, 4)
        print('Validation Rank-1 IR (Periocular)\t: ', peri_val_acc)    

        # print('Face')
        face_val_acc = identification.intramodal_id(model, face_loader_val, face_loader_test, device=device, peri_flag=False)
        face_val_acc = np.around(face_val_acc, 4)
        print('Validation Rank-1 IR (Face)\t: ', face_val_acc)    

        # Testing (Ethnic)
        ethnic_peri_test_acc = identification.intramodal_id(model, ethnic_peri_val_loader, ethnic_peri_test_loader, device=device, peri_flag=True)
        ethnic_peri_test_acc = np.around(ethnic_peri_test_acc, 4)
        print('Test Rank-1 IR (Ethnic - Periocular)\t: ', ethnic_peri_test_acc)   

        ethnic_face_test_acc = identification.intramodal_id(model, ethnic_face_val_loader, ethnic_face_test_loader, device=device, peri_flag=False)
        ethnic_face_test_acc = np.around(ethnic_face_test_acc, 4)
        print('Test Rank-1 IR (Ethnic - Face)\t: ', ethnic_face_test_acc)   
        
        if args.write_log is True:
            file = open(log_nm, 'a+')
            file.write(str('Epoch {}/{}'.format(epoch + 1, epochs)) + "\n")
            file.write('Loss : ' + str(loss) + "\n")
            file.write('Validation Rank-1 IR (Periocular)\t: ' + str(peri_val_acc) + "\n")
            file.write('Validation Rank-1 IR (Face)\t: ' + str(face_val_acc) + "\n")
            file.write('Test Rank-1 IR (Periocular) \t: ' + str(ethnic_peri_test_acc) + "\n")
            file.write('Test Rank-1 IR (Face) \t: ' + str(ethnic_face_test_acc) + "\n\n")
            file.close()

        if peri_val_acc >= peri_best_test_acc and epoch + 1 >= lr_sch[0] and save == True:           
            best_epoch = epoch + 1
            best_train_acc = train_acc

            peri_best_test_acc = peri_val_acc

            best_model = copy.deepcopy(model.state_dict())
            best_face_fc = copy.deepcopy(face_fc.state_dict())
            best_peri_fc = copy.deepcopy(peri_fc.state_dict())

            print('\n***** *****\n')
            print('Saving Best Model & Rank-1 IR ... ')
            print()
            
            # Set save_best_model_path
            tag = str(args.method) +  '/' + net_tag + '_' + str(batch_sub) + '_' + str(batch_samp) + '/'
            
            save_best_model_dir = './models/best_model/' + tag
            if not os.path.exists(save_best_model_dir):
                os.makedirs(save_best_model_dir)

            save_best_face_fc_dir = './models/best_face_fc/' + tag
            if not os.path.exists(save_best_face_fc_dir):
                os.makedirs(save_best_face_fc_dir)
                
            save_best_peri_fc_dir = './models/best_peri_fc/' + tag
            if not os.path.exists(save_best_peri_fc_dir):
                os.makedirs(save_best_peri_fc_dir)
                
            save_best_acc_dir = './models/best_acc/' + tag
            if not os.path.exists(save_best_acc_dir):
                os.makedirs(save_best_acc_dir)  
                    
            tag = str(args.method) + '_S' + str(cf_s) + '_M' + str(cf_m) + '_' + str(args.remarks) 

            save_best_model_path = save_best_model_dir + tag + '_' + str(start_string) + '.pth'
            save_best_face_fc_path = save_best_face_fc_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_peri_fc_path = save_best_peri_fc_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_acc_path = save_best_acc_dir + tag + '_' + str(start_string) + '.pkl' 
                    
            print('Best Model Pth\t: ', save_best_model_path)
            print('Best Face-FC Pth\t: ', save_best_face_fc_path)
            print('Best Peri-FC Pth\t: ', save_best_peri_fc_path)
            print('Best Rank-1 IR Pth\t: ', save_best_acc_path)

            # *** ***
            
            torch.save(best_model, save_best_model_path)
            torch.save(best_face_fc, save_best_face_fc_path)
            torch.save(best_peri_fc, save_best_peri_fc_path)

            with open(save_best_acc_path, 'wb') as f:
                pickle.dump([ best_epoch, best_train_acc, peri_best_test_acc, peri_best_pr_test_acc ], f)


    if args.write_log is True:
        file = open(log_nm, 'a+')
        end_ = datetime.now()
        end_string = end_.strftime("%Y%m%d_%H%M%S")
        file.write('Training completed at ' + str(end_) + ".\n\n")
        file.write("Model: " + str(save_best_model_path) + "\n\n")    
        file.close()

    # *** *** *** ***
        
    # Evaluation (Validation)
    print('**** Validation **** \n')
    # Identification - Validation
    print('Periocular - Validation')
    peri_val_acc = identification.intramodal_id(model, peri_loader_val, peri_loader_test, device=device, peri_flag=True)
    peri_val_acc = np.around(peri_val_acc, 4)
    print('Validation Rank-1 IR (Periocular)\t: ', peri_val_acc)

    print('Face - Validation')
    face_val_acc = identification.intramodal_id(model, face_loader_val, face_loader_test, device=device, peri_flag=False)
    face_val_acc = np.around(face_val_acc, 4)
    print('Validation Rank-1 IR (Face)\t: ', face_val_acc)

    # *** *** *** ***
    #### Identification and Verification for Test Datasets ( Ethnic, Pubfig, FaceScrub, IMDb Wiki, AR)

    print('\n**** Testing Evaluation (All Datasets) **** \n')
    #### Identification (Face and Periocular)
    print("Rank-1 IR (Periocular) \n")
    peri_id_dict = identification.im_id_main(model, root_pth=config.evaluation['identification'], modal='periocular', peri_flag=True, device=device)
    peri_id_dict = copy.deepcopy(peri_id_dict)
    print(peri_id_dict)

    print("Rank-1 IR (Face) \n")
    face_id_dict = identification.im_id_main(model, root_pth=config.evaluation['identification'], modal='face', peri_flag=False, device=device)
    face_id_dict = copy.deepcopy(face_id_dict)
    print(face_id_dict)

    #### Verification (Face and Periocular)
    print("EER (Periocular) \n")
    peri_eer_dict = verification.im_verify(model, out_features, root_drt=config.evaluation['verification'], peri_flag=True, device=device)
    peri_eer_dict = copy.deepcopy(peri_eer_dict)
    print(peri_eer_dict)

    print("EER (Face) \n")
    face_eer_dict = verification.im_verify(model, out_features, root_drt=config.evaluation['verification'], peri_flag=False, device=device)
    face_eer_dict = copy.deepcopy(face_eer_dict)
    print(face_eer_dict)

    #### Cross-modal Identification (Face and Periocular)
    print("Cross-Modal Rank-1 IR\n")
    cm_id_dict_f, cm_id_dict_p= identification.cm_id_main(model, root_pth=config.evaluation['identification'], device=device)
    cm_id_dict_f, cm_id_dict_p = copy.deepcopy(cm_id_dict_f), copy.deepcopy(cm_id_dict_p)
    print(cm_id_dict_p, cm_id_dict_f)

    #### Cross-modal Verification (Face and Periocular)
    print("Cross-Modal EER\n")
    cm_eer_dict = verification.cm_verify(model, face_model=None, peri_model=None, emb_size=out_features, root_drt=config.evaluation['verification'], device=device)    
    cm_eer_dict = copy.deepcopy(cm_eer_dict)
    print(cm_eer_dict)

    #### Multimodal Identification (Face and Periocular)
    print("Multimodal Rank-1 IR\n")
    mm_id_dict_concat = identification.mm_id_main(model, root_pth=config.evaluation['identification'], mode='concat', device=device)
    mm_id_dict_concat = copy.deepcopy(mm_id_dict_concat)
    print('Concat:', mm_id_dict_concat)

    #### Multimodal Verification (Face and Periocular)
    print("Multimodal EER\n")
    mm_eer_dict_concat = verification.mm_verify(model, face_model=None, peri_model=None, emb_size=out_features, root_drt=config.evaluation['verification'], mode='concat', device=device)
    mm_eer_dict_concat = copy.deepcopy(mm_eer_dict_concat)
    print('Concat:', mm_eer_dict_concat)

    # *** *** *** ***
    # Dataset Performance Summary
    print('**** Testing Summary Results (All Datasets) **** \n')

    # *** ***
    print('\n\n Ethnic \n')

    ethnic_acc_peri = peri_id_dict['ethnic']
    ethnic_eer_peri = peri_eer_dict['ethnic']
    ethnic_acc_face = face_id_dict['ethnic']
    ethnic_eer_face = face_eer_dict['ethnic']
    ethnic_cm_acc_p = cm_id_dict_p['ethnic']
    ethnic_cm_acc_f = cm_id_dict_f['ethnic']
    ethnic_cm_eer = cm_eer_dict['ethnic']
    ethnic_mm_acc_concat = mm_id_dict_concat['ethnic']
    ethnic_mm_eer_concat = mm_eer_dict_concat['ethnic']

    print('Rank-1 IR (Periocular)\t: ', ethnic_acc_peri)
    print("EER (Periocular)\t: ", ethnic_eer_peri)
    print('Rank-1 IR (Face)\t: ', ethnic_acc_face)
    print("EER (Face)\t: ", ethnic_eer_face)
    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', ethnic_cm_acc_p)
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', ethnic_cm_acc_f)
    print('Cross-modal EER \t: ', ethnic_cm_eer)
    print('Multimodal Rank-1 IR (Concat) \t: ', ethnic_mm_acc_concat)
    print('Multimodal EER (Concat)\t: ', ethnic_mm_eer_concat)


    # *** ***
    print('\n\n Pubfig \n')

    pubfig_acc_peri = peri_id_dict['pubfig']
    pubfig_eer_peri = peri_eer_dict['pubfig']
    pubfig_acc_face = face_id_dict['pubfig']
    pubfig_eer_face = face_eer_dict['pubfig']
    pubfig_cm_acc_p = cm_id_dict_p['pubfig']
    pubfig_cm_acc_f = cm_id_dict_f['pubfig']
    pubfig_cm_eer = cm_eer_dict['pubfig']
    pubfig_mm_acc_concat = mm_id_dict_concat['pubfig']
    pubfig_mm_eer_concat = mm_eer_dict_concat['pubfig']

    print('Rank-1 IR (Periocular)\t: ', pubfig_acc_peri)
    print("EER (Periocular)\t: ", pubfig_eer_peri)
    print('Rank-1 IR (Face)\t: ', pubfig_acc_face)
    print("EER (Face)\t: ", pubfig_eer_face)
    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', pubfig_cm_acc_p)
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', pubfig_cm_acc_f)
    print('Cross-modal EER \t: ', pubfig_cm_eer)
    print('Multimodal Rank-1 IR (Concat) \t: ', pubfig_mm_acc_concat)
    print('Multimodal EER (Concat)\t: ', pubfig_mm_eer_concat)


    # *** ***
    print('\n\n FaceScrub\n')

    facescrub_acc_peri = peri_id_dict['facescrub']
    facescrub_eer_peri = peri_eer_dict['facescrub']
    facescrub_acc_face = face_id_dict['facescrub']
    facescrub_eer_face = face_eer_dict['facescrub']
    facescrub_cm_acc_p = cm_id_dict_p['facescrub']
    facescrub_cm_acc_f = cm_id_dict_f['facescrub']
    facescrub_cm_eer = cm_eer_dict['facescrub']
    facescrub_mm_acc_concat = mm_id_dict_concat['facescrub']
    facescrub_mm_eer_concat = mm_eer_dict_concat['facescrub']

    print('Rank-1 IR (Periocular)\t: ', facescrub_acc_peri)
    print("EER (Periocular)\t: ", facescrub_eer_peri)
    print('Rank-1 IR (Face)\t: ', facescrub_acc_face)
    print("EER (Face)\t: ", facescrub_eer_face)
    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', facescrub_cm_acc_p)
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', facescrub_cm_acc_f)
    print('Cross-modal EER \t: ', facescrub_cm_eer)
    print('Multimodal Rank-1 IR (Concat) \t: ', facescrub_mm_acc_concat)
    print('Multimodal EER (Concat)\t: ', facescrub_mm_eer_concat)


    # *** *** *** ***
    print('\n\n IMDB Wiki \n')

    imdb_wiki_acc_peri = peri_id_dict['imdb_wiki']
    imdb_wiki_eer_peri = peri_eer_dict['imdb_wiki']
    imdb_wiki_acc_face = face_id_dict['imdb_wiki']
    imdb_wiki_eer_face = face_eer_dict['imdb_wiki']
    imdb_wiki_cm_acc_p = cm_id_dict_p['imdb_wiki']
    imdb_wiki_cm_acc_f = cm_id_dict_f['imdb_wiki']
    imdb_wiki_cm_eer = cm_eer_dict['imdb_wiki']
    imdb_wiki_mm_acc_concat = mm_id_dict_concat['imdb_wiki']
    imdb_wiki_mm_eer_concat = mm_eer_dict_concat['imdb_wiki']

    print('Rank-1 IR (Periocular)\t: ', imdb_wiki_acc_peri)
    print("EER (Periocular)\t: ", imdb_wiki_eer_peri)
    print('Rank-1 IR (Face)\t: ', imdb_wiki_acc_face)
    print("EER (Face)\t: ", imdb_wiki_eer_face)
    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', imdb_wiki_cm_acc_p)
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', imdb_wiki_cm_acc_f)
    print('Cross-modal EER \t: ', imdb_wiki_cm_eer)
    print('Multimodal Rank-1 IR (Concat) \t: ', imdb_wiki_mm_acc_concat)
    print('Multimodal EER (Concat)\t: ', imdb_wiki_mm_eer_concat)


    # *** *** *** ***
    print('\n\n AR \n')

    ar_acc_peri = peri_id_dict['ar']
    ar_eer_peri = peri_eer_dict['ar']
    ar_acc_face = face_id_dict['ar']
    ar_eer_face = face_eer_dict['ar']
    ar_cm_acc_p = cm_id_dict_p['ar']
    ar_cm_acc_f = cm_id_dict_f['ar']
    ar_cm_eer = cm_eer_dict['ar']
    ar_mm_acc_concat = mm_id_dict_concat['ar']
    ar_mm_eer_concat = mm_eer_dict_concat['ar']

    print('Rank-1 IR (Periocular)\t: ', ar_acc_peri)
    print("EER (Periocular)\t: ", ar_eer_peri)
    print('Rank-1 IR (Face)\t: ', ar_acc_face)
    print("EER (Face)\t: ", ar_eer_face)
    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', ar_cm_acc_p)
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', ar_cm_acc_f)
    print('Cross-modal EER \t: ', ar_cm_eer)
    print('Multimodal Rank-1 IR (Concat) \t: ', ar_mm_acc_concat)
    print('Multimodal EER (Concat)\t: ', ar_mm_eer_concat)

    # *** *** *** ***
    #### Average of all Datasets
    print('\n\n\n Calculating Average \n')

    avg_peri_ir = identification.get_avg(peri_id_dict)
    avg_face_ir = identification.get_avg(face_id_dict)
    avg_cm_p_ir = identification.get_avg(cm_id_dict_p)
    avg_cm_f_ir = identification.get_avg(cm_id_dict_f)
    avg_mm_ir_concat = identification.get_avg(mm_id_dict_concat)
    avg_peri_eer = verification.get_avg(peri_eer_dict)
    avg_face_eer = verification.get_avg(face_eer_dict)
    avg_cm_eer = verification.get_avg(cm_eer_dict)
    avg_mm_eer_concat = verification.get_avg(mm_eer_dict_concat)

    print('Rank-1 IR (Periocular)\t: ', avg_peri_ir['avg'])
    print('Rank-1 IR (Face)\t: ', avg_face_ir['avg'])
    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', avg_cm_p_ir['avg'])
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', avg_cm_f_ir['avg'])
    print('Multimodal Rank-1 IR (Concat) \t: ', avg_mm_ir_concat['avg'])
    print("EER (Periocular)\t: ", avg_peri_eer['avg'])
    print("EER (Face)\t: ", avg_face_eer['avg'])
    print('Cross-modal EER \t: ', avg_cm_eer['avg'])
    print('Multimodal EER (Concat)\t: ', avg_mm_eer_concat['avg'])


    # *** *** *** ***
    # Write Final Performance Summaries to Log 

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write('****Ethnic:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['ethnic']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['ethnic']))
        file.write('\n\nCross-Modal (ID): \n Periocular Gallery - ' + str(cm_id_dict_p['ethnic']) + ', \n Face Gallery - ' + str(cm_id_dict_f['ethnic']))
        file.write('\n\nMultimodal (ID): \n Concat - ' + str(mm_id_dict_concat['ethnic']) + '\n\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['ethnic']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ethnic']))        
        file.write('\n\nCross-Modal (Ver): ' + str(cm_eer_dict['ethnic']))
        file.write('\n\nMultimodal (Ver): \n Concat - ' + str(mm_eer_dict_concat['ethnic']) + '\n\n\n')
        file.write('****Pubfig:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['pubfig']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['pubfig']))
        file.write('\n\nCross-Modal (ID): \n Periocular Gallery - ' + str(cm_id_dict_p['pubfig']) + ', \n Face Gallery - ' + str(cm_id_dict_f['pubfig']))
        file.write('\n\nMultimodal (ID): \n Concat - ' + str(mm_id_dict_concat['pubfig']) + '\n\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['pubfig']) + '\nFinal EER. (Face): ' + str(face_eer_dict['pubfig']))        
        file.write('\n\nCross-Modal (Ver): ' + str(cm_eer_dict['pubfig']))
        file.write('\n\nMultimodal (Ver): \n Concat - ' + str(mm_eer_dict_concat['pubfig']) + '\n\n\n')
        file.write('****FaceScrub:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['facescrub']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['facescrub']))
        file.write('\n\nCross-Modal (ID): \n Periocular Gallery - ' + str(cm_id_dict_p['facescrub']) + ', \n Face Gallery - ' + str(cm_id_dict_f['facescrub']))
        file.write('\n\nMultimodal (ID): \n Concat - ' + str(mm_id_dict_concat['facescrub']) + '\n\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['facescrub']) + '\nFinal EER. (Face): ' + str(face_eer_dict['facescrub']))        
        file.write('\n\nCross-Modal (Ver): ' + str(cm_eer_dict['facescrub']))
        file.write('\n\nMultimodal (Ver): \n Concat - ' + str(mm_eer_dict_concat['facescrub']) + '\n\n\n')
        file.write('****IMDB Wiki:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['imdb_wiki']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['imdb_wiki']))
        file.write('\n\nCross-Modal (ID): \n Periocular Gallery - ' + str(cm_id_dict_p['imdb_wiki']) + ', \n Face Gallery - ' + str(cm_id_dict_f['imdb_wiki']))
        file.write('\n\nMultimodal (ID): \n Concat - ' + str(mm_id_dict_concat['imdb_wiki']) + '\n\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['imdb_wiki']) + '\nFinal EER. (Face): ' + str(face_eer_dict['imdb_wiki']))        
        file.write('\n\nCross-Modal (Ver): ' + str(cm_eer_dict['imdb_wiki']))
        file.write('\n\nMultimodal (Ver): \n Concat - ' + str(mm_eer_dict_concat['imdb_wiki']) + '\n\n\n')
        file.write('****AR:****')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(peri_id_dict['ar']) + '\nFinal Test Rank-1 IR (Face): ' + str(face_id_dict['ar']))
        file.write('\n\nCross-Modal (ID): \n Periocular Gallery - ' + str(cm_id_dict_p['ar']) + ', \n Face Gallery - ' + str(cm_id_dict_f['ar']))
        file.write('\n\nMultimodal (ID): \n Concat - ' + str(mm_id_dict_concat['ar']) + '\n\n')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['ar']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ar']))        
        file.write('\n\nCross-Modal (Ver): ' + str(cm_eer_dict['ar']))
        file.write('\n\nMultimodal (Ver): \n Concat - ' + str(mm_eer_dict_concat['ar']) + '\n\n\n')

        file.write('\n\n\n **** Average **** \n\n\n')
        file.write('\nFinal Test Rank-1 IR (Periocular): ' + str(avg_peri_ir['avg']) + '\nFinal Test Rank-1 IR (Face): ' + str(avg_face_ir['avg']))
        file.write('\n\nCross-Modal (ID): \n Periocular Gallery - ' + str(avg_cm_p_ir['avg']) + ', \n Face Gallery - ' + str(avg_cm_f_ir['avg']))
        file.write('\n\nMultimodal (ID): \n Concat - ' + str(avg_mm_ir_concat['avg']) + '\n\n')
        file.write('\nFinal EER. (Periocular): ' + str(avg_peri_eer['avg']) + '\nFinal EER. (Face): ' + str(avg_face_eer['avg']))        
        file.write('\n\nCross-Modal (Ver): ' + str(avg_cm_eer['avg']))
        file.write('\n\nMultimodal (Ver): \n Concat - ' + str(avg_mm_eer_concat['avg']) + '\n\n\n')
        file.close()

    # *** *** *** ***                 