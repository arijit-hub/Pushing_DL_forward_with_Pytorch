# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:38:28 2021

@author: Arijit
"""

import math
import torch

def lr_finder(model , loss_function , optimizer , start_lr , end_lr , train_dataloader):
    '''    
    Parameters
    ----------
    model : A pytorch model.
    loss_function : A pytorch loss function.
    optimizer : An alrady initialized optimizer function.
    start_lr : The starting learning rate.
    end_lr : The ending learning rate
    train_dataloader : The training dataloader.

    Returns
    -------
    Two lists, one with the learning rates and other with their equivalent losses.
    '''
    
    ## Getting the total steps in a batch ##
    total_steps = len(train_dataloader) - 1
    
    ## Setting each update of learning rate ##
    update_step = (end_lr / start_lr) ** (1 / total_steps)
    
    ## Setting a variable for learning rate ##
    lr = start_lr
    
    
    ## Initializing the learning and loss lists ##
    log_lr = []
    losses = []
    
    best_lr = 0.0
    best_loss = 0.0
    batch_num = 0
    ## Formulating our gradient form ##
    
    for x , y in train_dataloader:
        
        batch_num += 1
        
        ## Setting the initial learning rate ##
        optimizer.param_groups[0]['lr'] = lr
        
        ## Setting the gradient to zero ##
        optimizer.zero_grad()
        
        ## get predictions ##
        pred = model(x)
        
        ## Check Loss ##
        loss = loss_function(pred , y.reshape(-1 , 1).type(torch.cuda.FloatTensor))
        
        ## Calculate gradient ##
        loss.backward()
        
        ## Make one step ##
        optimizer.step()
        
        ## Store the log of the learning rate ##
        if (loss.item() < best_loss) or (batch_num == 1):
            best_loss = loss.item()
            best_lr = lr
            
        if (batch_num > 1) and (loss.item() > (best_loss * 4)) :
            break
        
        log_lr.append(math.log10(lr))
        
        ## Store the loss ##
        losses.append(loss.item())
        
        lr *= update_step
        
    return best_loss , best_lr , log_lr[10 : -5] , losses[10 : -5]