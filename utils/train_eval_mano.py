import time
import os
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import openmesh as om

DEEPCONTACT_BIN_WEIGHTS_FILE = 'data/class_bin_weights.out'
DEEPCONTACT_NUM_BINS = 10

def run(model, train_loader, train_loader_hoi, test_loader, test_loader_hoi, epochs, optimizer, scheduler, writer, meshdata,
        device):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss, train_l1, train_contact_loss, train_mano_l1 = train(model, optimizer, train_loader, train_loader_hoi, meshdata, device)
        t_duration = time.time() - t
        test_loss, test_l1, test_contact_loss, test_mano_l1 = test(model, test_loader, test_loader_hoi, epoch, meshdata, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration,
            'train_l1' : train_l1,
            'train_mano_l1' : train_mano_l1,
            'train_contact_loss' : train_contact_loss,
            'test_l1' : train_l1,
            'test_mano_l1' : test_mano_l1,
            'test_contact_loss' : train_contact_loss,
        }
        writer.print_info(info)
        print(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer,  train_loader, train_loader_hoi, meshdata, device):

    model.train()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0
    total_l1_mano_loss = 0

    # train_loss, train_l1, train_contact_loss, train_mano_l1

    for hoi, hand_mesh in zip( tqdm.tqdm(train_loader_hoi), train_loader):    
    
        optimizer.zero_grad()
        x = hand_mesh.x.to(device)

        out, verts = model(x)

        verts = verts / 1000
        
        mean = meshdata.mean.unsqueeze(0).to(device)
        std  = meshdata.std.unsqueeze(0).to(device)

        normalized_verts = (verts - mean) / std

        l1_loss = F.l1_loss(out[:,:,:3], x[:,:,:3], reduction='mean')
        l1_mano_loss = F.l1_loss(normalized_verts[:,:,:3], x[:,:,:3], reduction='mean')

        loss = l1_loss + l1_mano_loss 
        total_l1_loss += l1_loss
        total_l1_mano_loss += l1_mano_loss

        loss.backward()

        total_loss += loss
        optimizer.step()
        
    return total_loss / len(train_loader), total_l1_loss / len(train_loader), 0, total_l1_mano_loss / len(train_loader)

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


def test(model, test_loader, test_loader_hoi, epoch, meshdata, device):
    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0
    total_l1_mano_loss = 0

    rendering_first = True
    
    with torch.no_grad():
        for hoi, hand_mesh in zip( tqdm.tqdm(test_loader_hoi), test_loader): 
                
            x = hand_mesh.x.to(device)

            pred, mano_pred = model(x)

            mano_pred = mano_pred / 1000

            mean = meshdata.mean.unsqueeze(0).to(device)
            std  = meshdata.std.unsqueeze(0).to(device)

            normalized_verts = (mano_pred - mean) / std

            if rendering_first :

                ############### epoch ##############

                ########## Rendering Results #######

                # contact_hand.shape
                verts = pred[:,:,:3]                

                save_path = f'/scratch/minjay/coma/out/mesh_results/{epoch}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                hand_face = meshdata.template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = verts[hand_idx,:,:].cpu()
                    hand_mesh_verts = hand_mesh_verts * meshdata.std + meshdata.mean
 
                    om.write_mesh( save_path + f'verts_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))

                    hand_mesh_verts = mano_pred[hand_idx,:,:].cpu() 
                    
                    om.write_mesh( save_path + f'verts_mano_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))


                rendering_first = False

            l1_loss = F.l1_loss(pred[:,:,:3], x[:,:,:3], reduction='mean')
            l1_mano_loss = F.l1_loss(normalized_verts[:,:,:3], x[:,:,:3], reduction='mean')

            total_l1_loss += l1_loss
            #total_contact_loss = 0
            total_l1_mano_loss += l1_mano_loss
            
            total_loss = total_loss + l1_loss.item() + l1_mano_loss.item()
            
    return  total_loss / len(test_loader), total_l1_loss / len(test_loader), 0, total_l1_mano_loss / len(test_loader)


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
