import torch 

def update_model_breaking_symmetry(layers_to_break, actor, critic, maturity_threshold, colors, dev_, bounds,ages,num_new_features_to_replace,accumulated_num_features_to_replace):
    num_layers_to_break = len(layers_to_break)
    
    features_to_replace_indices = [None for _ in range(num_layers_to_break)]
    base_nodes_to_modify_indices = [None for _ in range(num_layers_to_break)]
    num_features_to_replace = [0 for _ in range(num_layers_to_break)]

    for idx_l in range(num_layers_to_break):
        ages[idx_l] += 1

        # ================================================
        
        eligible_feature_indices = torch.where(ages[idx_l] > maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0: continue
        
        #accumulated_num_features_to_replace[idx_l] += num_new_features_to_replace[idx_l]
     
        # Case when the number of features to be replaced is between 0 and 1.
        #num_new_features_to_replace_layer = int(accumulated_num_features_to_replace[idx_l])
        #accumulated_num_features_to_replace[idx_l] -= num_new_features_to_replace_layer
        #if num_new_features_to_replace_layer == 0: continue
        #print('num_new_features_to_replace_layer-->',num_new_features_to_replace_layer)
        # Find features to replace in the current layer
        
        colors_layer = colors[idx_l].to(dev_)
        idx_colors = torch.unique(colors_layer)
        
        nontrivial_fibers = {}
        
        for cc in idx_colors:
            mask = (colors_layer == cc)
            fiber = torch.where(mask)[0]
            
            if len(fiber) > 1:
              nontrivial_fibers[fiber[0]] = fiber[1:]

        num_nontrivial_fibers = len(nontrivial_fibers)

        if num_nontrivial_fibers == 0: continue
        
        lift_features = torch.cat(list(nontrivial_fibers.values()), dim=0)
        intersection = lift_features[torch.where(lift_features.unsqueeze(1) == eligible_feature_indices)[0]]
        LI = len(intersection)

        if LI == 0: continue
        # per_cover =0.1 and 0.5
        per_cover = 0.5
        num_new_features_to_replace_layer = 1 if LI < int(1./per_cover) else int(LI*per_cover) #1 if LI == 1 else LI // 2
        # new_features_to_replace = intersection[:num_new_features_to_replace_layer]
        new_features_to_replace = intersection[torch.randperm(len(intersection))[:num_new_features_to_replace_layer]]

        inverse_dict = {}
        
        for base_nodes, lift_nodes in nontrivial_fibers.items():
          for nn in lift_nodes:
              inverse_dict[nn.item()] = base_nodes
        
        base_nodes_to_modify = torch.tensor([inverse_dict[nn.item()] for nn in new_features_to_replace])
        
        num_features_to_replace[idx_l] = num_new_features_to_replace_layer
        features_to_replace_indices[idx_l] = new_features_to_replace
        base_nodes_to_modify_indices[idx_l] = base_nodes_to_modify

    #print('check inside braking-->',num_features_to_replace)

    #print(base_nodes_to_modify_indices)
    # ================================================

    with torch.no_grad():
    
        # Linear Layer ===================================
        if num_features_to_replace[0] != 0: 
            # Input weights
            bb = bounds[0]
            #print('features_to_replace_indices',features_to_replace_indices[0])
            #print('num_features_to_replace',num_features_to_replace[0])
            layers_to_break[0].bias.data[features_to_replace_indices[0]] *= 0.0
            layers_to_break[0].weight.data[features_to_replace_indices[0], :] = torch.empty(num_features_to_replace[0], layers_to_break[0].in_features, device = dev_).uniform_(-bb, bb)
            
            # Output weights.
            for id_base, id_lif in zip(base_nodes_to_modify_indices[0], features_to_replace_indices[0]):
                layers_to_break[1].weight_ih_l0.data[:, id_base] += layers_to_break[1].weight_ih_l0.data[:, id_lif]
            
            layers_to_break[1].weight_ih_l0.data[:, features_to_replace_indices[0]] = 0
            
            # Ages
            ages[0][features_to_replace_indices[0]] = 0

         # I rmeoved the breaking on the lstm
             # LSTM ===================================
        if num_features_to_replace[1] != 0: 
            hidden_size_lstm = layers_to_break[1].hidden_size
            
            # Input weights.
            bb = bounds[1]
            
            for kk in range(4):
                kk_features = hidden_size_lstm * kk + features_to_replace_indices[1]
                
                layers_to_break[1].bias_ih_l0.data[kk_features] *= 0.0
                layers_to_break[1].weight_ih_l0.data[kk_features, :] = torch.empty(num_features_to_replace[idx_l], layers_to_break[1].input_size , device = dev_).uniform_(-bb,bb)
                layers_to_break[1].bias_hh_l0.data[kk_features] *= 0.0
                layers_to_break[1].weight_hh_l0.data[kk_features, :] = torch.empty(num_features_to_replace[idx_l], layers_to_break[1].hidden_size, device = dev_).uniform_(-bb,bb)
                
            # Output weights.
            for id_base, id_lif in zip(base_nodes_to_modify_indices[1], features_to_replace_indices[1]):
                actor.weight.data[:, id_base] += actor.weight.data[:, id_lif]
                critic.weight.data[:, id_base] += critic.weight.data[:, id_lif]

            actor.weight.data[:, features_to_replace_indices[0]] = 0
            critic.weight.data[:, features_to_replace_indices[0]] = 0
            
            # Ages
            ages[1][features_to_replace_indices[0]] = 0 