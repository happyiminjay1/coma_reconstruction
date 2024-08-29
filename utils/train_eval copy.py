import time
import os
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import openmesh as om
from torch.utils.tensorboard import SummaryWriter
import pickle
import trimesh
from vedo import Points, show
from manopth.demo import display_hand
from manopth.manolayer import ManoLayer


DEEPCONTACT_BIN_WEIGHTS_FILE = 'data/class_bin_weights.out'
DEEPCONTACT_NUM_BINS = 10



def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    epoch_division_idx = 0

    train_load_size = len(train_loader)

    iteration_size = 200

    for epoch in range(1, epochs + 1):

        if iteration_size * (epoch_division_idx + 1) > train_load_size :

            epoch_division_idx = 0

        start_idx = epoch_division_idx * epoch_division_idx
        end_idx = min(train_load_size,start_idx+iteration_size)

        t = time.time()

        train_l1, train_contact_loss, train_taxonomy_loss = train(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device)
        
        t_duration = time.time() - t

        test_l1, f1_score, tax_acc = test(model, test_loader, epoch, loss_weight, 0, 30, device)

        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_l1' : train_l1 * dict_loss_weight['l1_loss'],
            'train_contact_loss' : train_contact_loss * dict_loss_weight['contact_loss'] ,
            'train_taxonomy_loss' : train_taxonomy_loss * dict_loss_weight['taxonomy_loss'],
            'test_l1' : test_l1 * dict_loss_weight['l1_loss'],
            'tax_acc' : tax_acc,
            'f1_score' : f1_score
        }

        writer.print_info(info)
        writer.s_writer(info,s_writer,epoch)
        print(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)
    
    s_writer.close()

def tester(model, train_loader, train_loader_hoi, test_loader, test_loader_hoi, epochs, optimizer, scheduler, writer, meshdata,exp_name, device, args):

    tester_env(model, test_loader, test_loader_hoi, meshdata,exp_name, device)


def train(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device):

    model.train()

    total_l1_loss = 0
    total_contact_loss = 0
    total_taxonomy_loss = 0

    # train_loss, train_l1, train_contact_loss, train_mano_l1

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    # 
    a = iter(train_loader)

    count_taken = 0

    for i in tqdm.tqdm(range(start_idx,end_idx)) :

        sample = next(a)
        count_taken += 1
        
        optimizer.zero_grad()

        x = sample['mano_verts']

        x_feature = sample['contact'].unsqueeze(-1)

        x = torch.cat((x,x_feature),dim=2)

        out, pred_taxonomy = model(x)

        contact_hand = out[:,:,3:13]

        gt_contact_map = val_to_class(x_feature).squeeze(2).long().to(device)

        bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
        criterion_taxonomy = torch.nn.CrossEntropyLoss()

        contact_classify_loss = criterion(contact_hand.permute(0, 2, 1), gt_contact_map)

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

        taxonomy_loss = criterion_taxonomy(pred_taxonomy,(gt_taxonomy - 1).to(device))

        total_contact_loss += contact_classify_loss
        total_taxonomy_loss += taxonomy_loss

        # pred 랑 x 랑 compare 해보기 

        l1_loss = F.l1_loss(out[:,:,:3], sample['mano_verts'], reduction='mean')
        total_l1_loss += l1_loss

        loss = l1_loss * dict_loss_weight['l1_loss'] + contact_classify_loss * dict_loss_weight['contact_loss'] + taxonomy_loss * dict_loss_weight['taxonomy_loss']
        
        loss.backward()

        optimizer.step()
        
    return total_l1_loss / count_taken , total_contact_loss / count_taken , total_taxonomy_loss / count_taken

def val_to_class(val):

    """

    Converts a contact value [0-1] to a class assignment

    :param val: tensor (batch, verts)

    :return: class assignment (batch, verts)

    """

    expanded = torch.floor(val * DEEPCONTACT_NUM_BINS)

    return torch.clamp(expanded, 0, DEEPCONTACT_NUM_BINS - 1).long() # Cut off potential 1.0 inputs?

def class_to_val(raw_scores):

    """

    Finds the highest softmax for each class

    :param raw_scores: tensor (batch, verts, classes)

    :return: highest class (batch, verts)

    """

    cls = torch.argmax(raw_scores, dim=2)

    val = (cls + 0.5) / DEEPCONTACT_NUM_BINS

    return val


def test(model, test_loader, epoch, loss_weight,start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    
    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0
    total_l1_mano_loss = 0
    total_acc_g = 0
    total_acc_c = 0
    total_acc_nc = 0

    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    total_taxonomy_loss = 0
    total_taxonomy_acr = 0

    #test_taxonomy_loss, taxonomy_acc

    rendering_first = True

    a = iter(train_loader)

    count_taken = 0
    
    with torch.no_grad():
        

        for i in tqdm.tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            count_taken += 1
            
            x = hand_mesh.x.to(device)
            x_feature = sample['contact'].unsqueeze(-1)

            # add input contact features
            x = torch.cat((x,x_feature),dim=2)

            pred, pred_taxonomy = model(x)

            contact_hand = pred[:,:,3:13]
            gt_contact_map = val_to_class(x_feature).squeeze(2).long().to(device)

            contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_gt_map = gt_contact_map.cpu().data.numpy()
            
            mask1 = contact_pred_map > 1
            mask2 = contact_gt_map > 1

            mask3 = contact_pred_map == 0
            mask4 = contact_gt_map == 0

            contact_pred_mask = contact_pred_map > 3
            contact_gt_mask = contact_gt_map > 3
            
            TP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == True))
            FP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == False))
            FN = np.sum(np.logical_and(contact_pred_mask == False, contact_gt_mask == True))

            #precision = (contact_pred_map[mask_TP_and_FP] == contact_gt_map[mask_TP_and_FP]).mean()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1_score = 2 * (precision * recall) / (precision + recall)

            f1_score = torch.tensor(f1_score,dtype=torch.float32)

            total_f1_score += f1_score

            #total_precision, total_recall

            # mask_or_c = np.logical_or(mask1, mask2)
            # mask_or_nc = np.logical_or(mask3, mask4)

            # acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            # acc_c = (contact_pred_map[mask_or_c] == contact_gt_map[mask_or_c]).mean()
            # acc_nc = (contact_pred_map[mask_or_nc] == contact_gt_map[mask_or_nc]).mean()
            
            # acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            # total_acc_g += acc_g 

            # total_acc_c += acc_c
            # total_acc_nc += acc_nc
            gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == (gt_taxonomy-1).numpy()).mean()

            total_taxonomy_acr += acc_taxonomy

            if rendering_first :

                ############### epoch ##############

                ########## Rendering Results #######
                verts = pred[:,:,:3]                

                save_path = f'/scratch/minjay/coma_taxonomy_prediction/out/{exp_name}/mesh_results/{epoch}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                import openmesh as om

                tmp_mesh = om.read_trimesh('/scratch/minjay/coma_taxonomy_prediction/template/hand_mesh_template.obj')
                template_face = tmp_mesh.face_vertex_indices()
                
                hand_face = template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = verts[hand_idx,:,:].cpu()
 
                    om.write_mesh( save_path + f'verts_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))

                rendering_first = False

            l1_loss = F.l1_loss(pred[:,:,:3], sample['mano_verts'], reduction='mean')

            total_l1_loss += l1_loss
            
            total_loss = l1_loss * dict_loss_weight['l1_loss'] + l1_mano_loss * dict_loss_weight['l1_mano_loss'] + contact_classify_loss * dict_loss_weight['contact_classify_loss'] + taxonomy_loss * dict_loss_weight['taxonomy_loss']
            
    return  total_l1_loss / count_taken, total_f1_score / count_taken, total_taxonomy_acr / count_taken


def tester_env(model, test_loader, test_loader_hoi, meshdata, exp_name, device):

    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0

    total_l1_mano_loss = 0
    total_l1_mano_loss_before = 0

    total_acc_refine = 0
    total_acc_origin = 0

    total_taxonomy_loss = 0
    total_taxonomy_acr = 0

    #test_taxonomy_loss, taxonomy_acc

    rendering_first = True

    all_data = []
    
    with torch.no_grad():
        for hoi, hand_mesh in zip( tqdm.tqdm(test_loader_hoi), test_loader): 
            
            x = hand_mesh.x.to(device)
            x_feature = hoi[1].float().to(device)

            # add input contact features
            x = torch.cat((x,x_feature),dim=2)

            pred, pred_taxonomy, mano_pred = model(x)

            contact_hand = pred[:,:,3:13]
            gt_contact_map = val_to_class(hoi[2]).squeeze(2).long().to(device)
            baseline_contact_map = val_to_class(hoi[9]).squeeze(2).long().to(device)

            bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
            criterion_taxonomy = torch.nn.CrossEntropyLoss()

            taxonomy_loss = criterion_taxonomy(pred_taxonomy,(hoi[3] -1).to(device))
            total_taxonomy_loss += taxonomy_loss

            contact_classify_loss = criterion(contact_hand.permute(0, 2, 1), gt_contact_map)
            total_contact_loss += contact_classify_loss

            contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_gt_map = gt_contact_map.cpu().data.numpy()

            #print(contact_pred_map[0])
            #print(contact_gt_map[0])


            acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            acc_basline = (gt_contact_map.cpu().data.numpy() == baseline_contact_map.cpu().data.numpy()).mean()

            total_acc_refine += acc_g 
            total_acc_origin += acc_basline 

            acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == (hoi[3]-1).numpy()).mean()
            total_taxonomy_acr += acc_taxonomy

            mano_pred = mano_pred / 1000

            mean = meshdata.mean.unsqueeze(0).to(device)
            std  = meshdata.std.unsqueeze(0).to(device)

            normalized_verts = (mano_pred - mean) / std

            hand_mesh_gt_verts = hoi[0].to(device)

            hand_verts_gt_normalized = (hand_mesh_gt_verts - mean) / std

            hand_mesh_pred_verts = hoi[6].to(device)

            hand_verts_pred_normalized = (hand_mesh_pred_verts - mean) / std


            if rendering_first :

                ############### epoch ##############

                ########## Rendering Results #######

                # contact_hand.shape

                
                verts = pred[:,:,:3]  
        

                save_path = f'/scratch/minjay/coma_refine_test/out/{exp_name}/mesh_results/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                hand_face = meshdata.template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = verts[hand_idx,:,:].cpu()
                    hand_mesh_verts = hand_mesh_verts * meshdata.std + meshdata.mean
 
                    om.write_mesh( save_path + f'verts_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))

                    hand_mesh_verts = mano_pred[hand_idx,:,:].cpu() 
                    
                    om.write_mesh( save_path + f'verts_mano_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))


                save_path = f'/scratch/minjay/coma_refine_test/out/{exp_name}/mesh_results/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                hand_face = meshdata.template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = hand_mesh_gt_verts[hand_idx,:,:].cpu()
                    #hand_mesh_verts = hand_mesh_verts * meshdata.std + meshdata.mean
 
                    om.write_mesh( save_path + f'verts_{hand_idx}_gt.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))


                rendering_first = False

                for hand_idx in range(verts.shape[0]) :

                    mesh_coloring = trimesh.load(save_path + f'verts_{hand_idx}.obj')

                    pc2 = Points(mesh_coloring.vertices, r=5)
                    pc2.cmap("gray", contact_pred_map[hand_idx])


                    mesh_coloring2 = trimesh.load(save_path + f'verts_{hand_idx}_gt.obj')

                    pc1 = Points(mesh_coloring2.vertices, r=5)
                    pc1.cmap("gray", contact_gt_map[hand_idx])


                    # Draw result on N=2 sync'd renderers
                    show([(mesh_coloring,pc2),(mesh_coloring2,pc1)], N=2, axes=1).close()
            
            exit(0)



            l1_loss = F.l1_loss(pred[:,:,:3], hand_verts_gt_normalized, reduction='mean')
            l1_mano_loss = F.l1_loss(normalized_verts[:,:,:3], hand_verts_gt_normalized, reduction='mean')

            l1_mano_loss_before = F.l1_loss(hand_verts_pred_normalized, hand_verts_gt_normalized, reduction='mean')

            total_l1_loss += l1_loss
            total_l1_mano_loss += l1_mano_loss

            total_l1_mano_loss_before += l1_mano_loss_before
            
            total_loss = total_loss + l1_loss.item() + l1_mano_loss.item() + contact_classify_loss.item() + taxonomy_loss.item()

            # print( hoi[10] )

            for idx, i in enumerate(hoi[10]) :
                
                new_sample = {}

                #new_sample['baseline_contact']
                print(baseline_contact_map[idx].cpu().data.numpy()[:])
                print(contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)[idx][:])

            exit(0)
                


        all_data.append(new_sample)

    #self.data[idx]['hand_mesh_gt'].x, self.data[idx]['hoi_feature'], self.data[idx]['contactmap'], 
    #self.data[idx]['taxonomy'],self.data[idx]['trans_gt'], self.data[idx]['rep_gt'], self.data[idx]['hand_mesh_pred'].x, 
    #self.data[idx]['trans_pred'], self.data[idx]['rep_pred'], self.data[idx]['contactmap_pred']

    #trans_gt = data_info['trans_gt']
    #rep_gt = data_info['rep_gt']

    #trans_pred = data_info['trans_pred']
    #rep_pred = data_info['rep_pred']

    print(total_acc_refine / len(test_loader), total_acc_origin/ len(test_loader))
    print(total_l1_mano_loss  / len(test_loader) , total_l1_mano_loss_before  / len(test_loader))
