import torch
import torch.nn as nn
import copy
import pickle
import pandas as pd
import re
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def collapse_linear_layer(model, clusters):
	
	num_clusters = torch.unique(clusters).shape[0]
	num_nodes = clusters.shape[0]

	matrix_partition = torch.zeros(num_clusters, num_nodes)
	matrix_partition.scatter_(0, clusters.unsqueeze(0), 1)
	sizes_clusters = torch.mm(matrix_partition, torch.ones(num_nodes, 1))

	W1 = model.policy.policy.network[7].weight.data
	W1_coll = torch.mm(matrix_partition, W1) / sizes_clusters.view(-1, 1)
	b1_coll = torch.mm(matrix_partition, model.policy.policy.network[7].bias.data.unsqueeze(1)) / sizes_clusters.view(-1, 1)

	collapsed_model = copy.deepcopy(model)
	collapsed_model.policy.recurrent = nn.LSTM(input_size=num_clusters, hidden_size=model.policy.recurrent.hidden_size)

	W2 = model.policy.recurrent.weight_ih_l0.data
	W2_coll = torch.mm(W2, matrix_partition.T)
	collapsed_model.policy.recurrent.weight_ih_l0.data = W2_coll
	collapsed_model.policy.recurrent.weight_hh_l0.data = model.policy.recurrent.weight_hh_l0.data

	collapsed_model.policy.recurrent.bias_ih_l0.data = model.policy.recurrent.bias_ih_l0.data
	collapsed_model.policy.recurrent.bias_hh_l0.data = model.policy.recurrent.bias_hh_l0.data

	collapsed_model.policy.policy.network[7] = nn.Linear(collapsed_model.policy.policy.network[7].in_features, num_clusters)
	collapsed_model.policy.policy.network[7].weight.data = W1_coll
	collapsed_model.policy.policy.network[7].bias.data = b1_coll.squeeze(1)
	
	return collapsed_model

def collapse_lstm(model, prev_layer_clusters, lstm_clusters):
	
	in_dim = model.policy.recurrent.input_size
	h_dim = model.policy.recurrent.hidden_size
	
	Whi, Whf, Whg, Who = model.policy.recurrent.weight_hh_l0.data[:h_dim,:], model.policy.recurrent.weight_hh_l0.data[h_dim:2*h_dim,:], model.policy.recurrent.weight_hh_l0.data[2*h_dim:3*h_dim,:], model.policy.recurrent.weight_hh_l0.data[3*h_dim:,:]
	Wii, Wif, Wig, Wio = model.policy.recurrent.weight_ih_l0.data[:h_dim,:], model.policy.recurrent.weight_ih_l0.data[h_dim:2*h_dim,:], model.policy.recurrent.weight_ih_l0.data[2*h_dim:3*h_dim,:], model.policy.recurrent.weight_ih_l0.data[3*h_dim:,:]

	bhi, bhf, bhg, bho = model.policy.recurrent.bias_hh_l0.data[:h_dim], model.policy.recurrent.bias_hh_l0.data[h_dim:2*h_dim], model.policy.recurrent.bias_hh_l0.data[2*h_dim:3*h_dim], model.policy.recurrent.bias_hh_l0.data[3*h_dim:]
	bii, bif, big, bio = model.policy.recurrent.bias_ih_l0.data[:h_dim], model.policy.recurrent.bias_ih_l0.data[h_dim:2*h_dim], model.policy.recurrent.bias_ih_l0.data[2*h_dim:3*h_dim], model.policy.recurrent.bias_ih_l0.data[3*h_dim:]

	# Num of clusters, Matrix of partition and sizes of the clusters
	num_in_clusters = torch.unique(prev_layer_clusters).shape[0]
	matrix_in_partition = torch.zeros(num_in_clusters, in_dim)
	matrix_in_partition.scatter_(0, prev_layer_clusters.unsqueeze(0), 1)
	sizes_in_clusters = torch.mm(matrix_in_partition, torch.ones(in_dim, 1))

	num_h_clusters = torch.unique(lstm_clusters).shape[0]
	matrix_h_partition = torch.zeros(num_h_clusters, h_dim)
	matrix_h_partition.scatter_(0, lstm_clusters.unsqueeze(0), 1)
	sizes_h_clusters = torch.mm(matrix_h_partition, torch.ones(h_dim, 1))

	dict_w = {'hi': Whi, 'hf': Whf, 'hg': Whg, 'ho': Who,
				'ii': Wii, 'if': Wif, 'ig': Wig, 'io':Wio}

	for name, W in dict_w.items():
		W_collap = torch.mm(matrix_h_partition, W) / sizes_h_clusters.view(-1, 1)

		partition = matrix_h_partition if 'h' in name else matrix_in_partition
		dict_w[name] = torch.mm(W_collap, partition.T)

	weight_hh_l0_collap = torch.cat([dict_w['hi'], dict_w['hf'], dict_w['hg'], dict_w['ho']])
	weight_ih_l0_collap = torch.cat([dict_w['ii'], dict_w['if'], dict_w['ig'], dict_w['io']])
	
	dict_bias = {'hi': bhi, 'hf': bhf, 'hg': bhg, 'ho': bho,
			'ii': bii, 'if': bif, 'ig': big, 'io':bio}
	
	for name, b in dict_bias.items():
		bias_collap = matrix_h_partition @ b / sizes_h_clusters
		dict_bias[name] = bias_collap[0, :]

	bias_hh_l0_collap = torch.cat([dict_bias['hi'], dict_bias['hf'], dict_bias['hg'], dict_bias['ho']])
	bias_ih_l0_collap = torch.cat([dict_bias['ii'], dict_bias['if'], dict_bias['ig'], dict_bias['io']])
	
	# Copy of the agent
	collapsed_model = copy.deepcopy(model)
	collapsed_model.policy.recurrent = nn.LSTM(num_in_clusters,num_h_clusters)
	collapsed_model.policy.recurrent.weight_hh_l0.data = weight_hh_l0_collap
	collapsed_model.policy.recurrent.weight_ih_l0.data = weight_ih_l0_collap
	collapsed_model.policy.recurrent.bias_hh_l0.data = bias_hh_l0_collap
	collapsed_model.policy.recurrent.bias_ih_l0.data = bias_ih_l0_collap
	
	collapsed_model.policy.policy.actor = nn.Linear(num_h_clusters, model.policy.policy.actor.out_features)
	W_actor = model.policy.policy.actor.weight.data
	W_actor_coll = torch.mm(W_actor, matrix_h_partition.T)
	collapsed_model.policy.policy.actor.weight.data = W_actor_coll
	collapsed_model.policy.policy.actor.bias.data = model.policy.policy.actor.bias.data
	
	collapsed_model.policy.policy.value_fn = nn.Linear(num_h_clusters, model.policy.policy.value_fn.out_features)
	W_value = model.policy.policy.value_fn.weight.data
	W_value_coll = torch.mm(W_value, matrix_h_partition.T)
	collapsed_model.policy.policy.value_fn.weight.data = W_value_coll
	collapsed_model.policy.policy.value_fn.bias.data = model.policy.policy.value_fn.bias.data

	return collapsed_model

def fibration_linear(weights, in_clusters, threshold, first_layer = False, bias=None):
	dim_out, dim_in  = weights.shape

	if first_layer:
		collapse_weights = weights
	else:	
		num_in_clusters  = len(np.unique(in_clusters))
		collapse_weights = torch.zeros((dim_out, num_in_clusters))

		for color in in_clusters: 
			indices_k = np.where(in_clusters == color)[0]
			collapse_weights[:, color] = weights[:, indices_k].sum(axis=1)

	if bias is not None:
		collapse_weights = torch.cat((collapse_weights, bias.unsqueeze(1)), dim=1)

	collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=1)
	distance = 1 - torch.mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
		n_clusters=None,
		distance_threshold=threshold,
		linkage='average',
		metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

def fibration_conv2d(weights, in_clusters, threshold, first_layer = False, bias=None):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if first_layer:
		collapse_weights = weights.view(out_n, -1)
	else:
		num_in_clusters  = len(np.unique(in_clusters))
		collapse_weights = torch.zeros((out_n, num_in_clusters, hx*hy))

		for color in in_clusters: 
			indices_k = np.where(in_clusters == color)[0]
			collapse_weights[:, color, :] = weights[:, indices_k, :].sum(axis=1)

		collapse_weights = collapse_weights.view(out_n,-1)

	if bias is not None:
		collapse_weights = torch.cat((collapse_weights, bias.unsqueeze(1)), dim=1)

	collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=1)
	distance = 1 - torch.mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
		n_clusters=None,
		distance_threshold=threshold,
		linkage='average',
		metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

def fibration_lstm(layer, in_clusters, threshold, T=5):
	input_size = layer.input_size
	hidden_size = layer.hidden_size
	weight_ih = layer.weight_ih_l0.detach()
	weight_hh = layer.weight_hh_l0.detach()
 
	h_clusters  = np.array([0  for idx in range(hidden_size)])
	c_clusters  = np.array([0  for idx in range(hidden_size)])
 
	for t in range(T):

		# (1) Collapse W_ii,...W_io based on in_cluster.

		dim_out, dim_in = weight_ih.shape
		num_in_clusters = len(np.unique(in_clusters))
		collapse_weights_i = torch.zeros((dim_out, num_in_clusters))

		for color in in_clusters:
			indices_k = np.where(in_clusters == color)[0]
			collapse_weights_i[:, color] = weight_ih[:, indices_k].sum(axis=1)

		# (2) Collapse W_hi,...W_ho based on h_cluster.

		dim_out, dim_in = weight_hh.shape
		num_h_clusters = len(np.unique(h_clusters))
		collapse_weights_h = torch.zeros((dim_out, num_h_clusters))

		for color in h_clusters:
			indices_k = np.where(h_clusters == color)[0]
			collapse_weights_h[:, color] = weight_hh[:, indices_k].sum(axis=1)

		collapse_weights = torch.cat((collapse_weights_i,collapse_weights_h),dim=1)

		# (3) Clusters of the gates

		collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=1)
		distance = 1 - torch.mm(collapse_weights_norm, collapse_weights_norm.T)
		distance = distance.cpu().numpy()

		clustering = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=threshold,
			linkage='average',
			metric='precomputed')

		gates_clusters = clustering.fit_predict(distance) # I, F, G, O

		# (4) (F,C,I,G) clusters
		gates_cluster_matrix = gates_clusters.reshape(4,hidden_size)

		# (5) c_clusters
		new_c_clusters = np.vstack((gates_cluster_matrix[:3,:], c_clusters))
		_, c_clusters = np.unique(new_c_clusters.T, axis=0, return_inverse=True)

		# (6) h_clusters
		new_h_clusters = np.vstack((gates_cluster_matrix[3,:], c_clusters))
		_, h_clusters = np.unique(new_h_clusters.T, axis=0, return_inverse=True)
  
	return h_clusters

def opfibration_linear(weights, out_clusters, threshold, last_layer = False, bias=None):
	dim_out, dim_in  = weights.shape

	if last_layer:
		collapse_weights = weights
	else:
		num_out_clusters = len(np.unique(out_clusters))
		collapse_weights = torch.zeros((num_out_clusters, dim_in))

		for color in out_clusters: 
			indices_k  = np.where(out_clusters == color)[0]
			collapse_weights[color,:] = weights[indices_k,:].sum(axis=0)

	if bias is not None:
		collapse_weights = torch.cat((collapse_weights, bias.unsqueeze(1)), dim=1)
  
	collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=0)
	distance = 1 - torch.mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
		n_clusters=None,
		distance_threshold=threshold,
		linkage='average',
		metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

def opfibration_conv2d(weights, out_clusters, threshold, last_layer = False, bias=None):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if last_layer:
		collapse_weights = weights.permute(0, 2, 1).reshape(-1, in_n)
	else:
		num_out_clusters = len(np.unique(out_clusters))	
		collapse_weights = torch.zeros((num_out_clusters, in_n, hx*hy))

		for color in out_clusters: 
			indices_k = np.where(out_clusters == color)[0]
			collapse_weights[color, :, :] = weights[indices_k, :, :].sum(axis=0)

		collapse_weights = collapse_weights.permute(0, 2, 1).reshape(-1, in_n)
  
	if bias is not None:
		collapse_weights = torch.cat((collapse_weights, bias.unsqueeze(1)), dim=1)

	collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=0)
	distance = 1 - torch.mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
		n_clusters=None,
		distance_threshold=threshold,
		linkage='average',
		metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

def opfibration_lstm(weight_ih, weight_hh, out_clusters, threshold, T=5):
	
	input_size = weight_ih.shape[1]
	in_clusters  = np.array([idx  for idx in range(input_size)])
	h_clusters = out_clusters
	dim_h = len(out_clusters)

	for t in range(T):

		# (1) Collapse W_ii,...W_io based on in_cluster.

		dim_out, dim_in  = weight_ih.shape
		num_in_clusters = len(np.unique(in_clusters))
		collapse_weights_i = torch.zeros((dim_out, num_in_clusters))

		for color in in_clusters:
			indices_k = np.where(in_clusters == color)[0]
			collapse_weights_i[:, color] = weight_ih[:, indices_k].sum(axis=1)		

		# (2) Collapse W_hi,...W_ho based on h_cluster.

		dim_out, dim_in  = weight_hh.shape
		num_h_clusters  = len(np.unique(h_clusters))
		collapse_weights_h = torch.zeros((dim_out, num_h_clusters))

		for color in h_clusters:
			indices_k = np.where(h_clusters == color)[0]
			collapse_weights_h[:, color] = weight_hh[:, indices_k].sum(axis=1)

		collapse_weights = torch.cat((collapse_weights_i,collapse_weights_h),dim=1)

		# (3) Clusters Fibration of the gates

		collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=1)
		distance = 1 - torch.mm(collapse_weights_norm, collapse_weights_norm.T)
		distance = distance.cpu().numpy()

		clustering = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=threshold,
			linkage='average',
			metric='precomputed')

		gates_fib_clusters = clustering.fit_predict(distance) # I, F, G, O
		gates_fib_cluster_matrix = gates_fib_clusters.reshape(4,hidden_size)

		# (4) c_clusters, gates_op_clusters
		c_clusters = np.vstack((gates_fib_cluster_matrix[[1,3],:], h_clusters))
		_, c_clusters = np.unique(c_clusters.T, axis=0, return_inverse=True)

		i_clusters = np.vstack((gates_fib_cluster_matrix[[1,2,3],:], h_clusters))
		_, i_clusters = np.unique(i_clusters.T, axis=0, return_inverse=True)

		f_clusters = np.vstack((gates_fib_cluster_matrix, h_clusters))
		_, f_clusters = np.unique(f_clusters.T, axis=0, return_inverse=True)

		g_clusters = np.vstack((gates_fib_cluster_matrix[[0,1,3],:], h_clusters))
		_, g_clusters = np.unique(g_clusters.T, axis=0, return_inverse=True)

		o_clusters = np.vstack((gates_fib_cluster_matrix[[0,1,2],:], h_clusters))
		_, o_clusters = np.unique(o_clusters.T, axis=0, return_inverse=True)

		dict_clusters = {'i':i_clusters,'f':f_clusters,'g':g_clusters,'o':o_clusters}

		# (5) gates_op_clusters

		Ws = {'i':torch.cat((weight_ih[:dim_h,:], weight_hh[:dim_h,:]), dim=1),
			'f':torch.cat((weight_ih[dim_h:2*dim_h,:], weight_hh[dim_h:2*dim_h,:]), dim=1),
			'g':torch.cat((weight_ih[2*dim_h:3*dim_h,:], weight_hh[2*dim_h:3*dim_h,:]), dim=1),
			'o':torch.cat((weight_ih[3*dim_h:4*dim_h,:], weight_hh[3*dim_h:4*dim_h,:]), dim=1)}

		all_collapse_weights = []

		for kk in ['i','f','g','o']:
			weights = Ws[kk]
			clusters = dict_clusters[kk]

			dim_out, dim_in  = weights.shape
			num_clusters = len(np.unique(clusters))
			collapse_weights = torch.zeros((num_clusters, dim_in))

			for color in clusters: 
				indices_k  = np.where(clusters == color)[0]
				collapse_weights[color,:] = weights[indices_k,:].sum(axis=0)

			all_collapse_weights.append(collapse_weights)

		all_collapse_weights = torch.cat(all_collapse_weights,dim=0)
		collapse_weights_norm = torch.nn.functional.normalize(all_collapse_weights, dim=0)
		distance = 1 - torch.mm(collapse_weights_norm.T, collapse_weights_norm)

		clustering = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=threshold,
			linkage='average',
			metric='precomputed')

		xh_clusters = clustering.fit_predict(distance)

		x_clusters = xh_clusters[:input_size]
		h_clusters = xh_clusters[input_size:]

		unique_values = np.unique(x_clusters)
		mapping = {val: idx for idx, val in enumerate(unique_values)}
		x_clusters = np.array([mapping[val] for val in x_clusters])

		unique_values = np.unique(h_clusters)
		mapping = {val: idx for idx, val in enumerate(unique_values)}
		h_clusters = np.array([mapping[val] for val in h_clusters])

	return x_clusters, h_clusters


def save_dict(data, filename):
	with open(filename, 'wb') as f:
		pickle.dump(data, f)
		
def load_dict(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)
	
def parse_pufferlib_log(filepath):
	# Read the file
	with open(filepath, 'r', encoding='utf-8') as f:
		text = f.read()
	
	text = text.replace('\x1b[0;0H', '').replace('\n\n', '\n')
	
	# Split into chunks based on the border pattern
	blocks = re.split(r'╯', text, flags=re.DOTALL)
	blocks = [block.strip() for block in blocks if block.strip()]

	records = []

	for i, block in enumerate(blocks):
		record = {}
		
		# Extract fields using regex patterns
		epoch = re.search(r'Epoch\s+(\d+)', block)
		agent_steps = re.search(r'Agent Steps\s+([\d.k]+)', block)
		sps = re.search(r'SPS\s+([\d.k]+)', block)
		uptime = re.search(r'Uptime\s+([\dhms\s]+)', block)
		remaining = re.search(r'Remaining\s+([\dhms\s]+)', block)
		policy_loss = re.search(r'policy_loss\s+(-?[\d.]+)', block)
		value_loss = re.search(r'value_loss\s+(-?[\d.]+)', block)
		entropy = re.search(r'entropy\s+(-?[\d.]+)', block)
		old_approx_kl = re.search(r'old_approx_kl\s+(-?[\d.]+)', block)
		approx_kl = re.search(r'approx_kl\s+(-?[\d.]+)', block)
		clipfrac = re.search(r'clipfrac\s+(-?[\d.]+)', block)
		explained_var = re.search(r'explained_var\S*\s+(-?[\d.]+)', block)
		episode_return = re.search(r'episode_return\s+(-?[\d.]+)', block)
		episode_length = re.search(r'episode_length\s+(-?[\d.]+)', block)

		# Fill the record
		record['epoch'] = i
		record['agent_steps'] = agent_steps.group(1) if agent_steps else None
		record['sps'] = sps.group(1) if sps else None
		record['uptime'] = uptime.group(1).strip() if uptime else None
		record['remaining'] = remaining.group(1).strip() if remaining else None
		record['policy_loss'] = float(policy_loss.group(1)) if policy_loss else None
		record['value_loss'] = float(value_loss.group(1)) if value_loss else None
		record['entropy'] = float(entropy.group(1)) if entropy else None
		record['old_approx_kl'] = float(old_approx_kl.group(1)) if old_approx_kl else None
		record['approx_kl'] = float(approx_kl.group(1)) if approx_kl else None
		record['clipfrac'] = float(clipfrac.group(1)) if clipfrac else None
		record['explained_var'] = float(explained_var.group(1)) if explained_var else None
		record['episode_return'] = float(episode_return.group(1)) if episode_return else None
		record['episode_length'] = float(episode_length.group(1)) if episode_length else None
		
		records.append(record)
	
	# Build the dataframe
	df = pd.DataFrame(records)

	# Optionally convert agent_steps and sps that contain 'k'
	def parse_number(x):
		if isinstance(x, str):
			if 'k' in x:
				return float(x.replace('k', '')) * 1000
			else:
				return float(x)
		return x

	df['agent_steps'] = df['agent_steps'].apply(parse_number)
	df['sps'] = df['sps'].apply(parse_number)

	return df

def sort_by_last_array(arrays):
	if not arrays or not all(len(sub) == len(arrays[0]) for sub in arrays):
		raise ValueError("All sub-arrays must be of the same length")

	transposed = list(zip(*arrays))

	transposed.sort(key=lambda x: x[-1])
	
	sorted_arrays = [list(x) for x in zip(*transposed)]
	return sorted_arrays


def sort_dict_by_key(d, sort_key):
	"""
	Sorts each list in the dictionary based on the ascending order of the values
	in the list corresponding to sort_key.

	Args:
		d (dict): Dictionary of lists with equal length.
		sort_key (str): Key whose list will be used to determine the sorting order.

	Returns:
		dict: New dictionary with all lists sorted based on sort_key's list.
	"""
	if sort_key not in d:
		raise ValueError(f"Key '{sort_key}' not found in dictionary.")
	
	sort_order = sorted(range(len(d[sort_key])), key=lambda i: d[sort_key][i])
	
	return {k: [v[i] for i in sort_order] for k, v in d.items()}


def reinitialize_weights(model):
	"""
	Reinitialize all weights in the model to random values using standard initialization
	methods depending on the layer type.
	"""
	for layer in model.modules():
		if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
			nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)
		elif isinstance(layer, (nn.Linear)):
			nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
			if layer.bias is not None:
				fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
				bound = 1 / math.sqrt(fan_in)
				nn.init.uniform_(layer.bias, -bound, bound)
		elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
			for name, param in layer.named_parameters():
				if 'weight_ih' in name:
					nn.init.xavier_uniform_(param.data)
				elif 'weight_hh' in name:
					nn.init.orthogonal_(param.data)
				elif 'bias' in name:
					nn.init.zeros_(param.data)
		elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
			nn.init.ones_(layer.weight)
			nn.init.zeros_(layer.bias)
			
			
def zero_all_biases(model):
	for module in model.modules():
		if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
			if module.bias is not None:
				nn.init.constant_(module.bias, 0)
		if isinstance(module, nn.LSTM):
			if module.bias is not None:
				nn.init.constant_(module.bias_ih_l0, 0)
				

def number_of_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_fibers_vs_time(pt_files, threshold, has_epoch_zero=False):
	
	model = torch.load(pt_files[0], weights_only=False, map_location='cpu')
	
	if hasattr(model.policy, 'policy'):
		layers = [layer for layer in model.policy.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))] + [model.policy.recurrent]
	else:
		layers = [layer for layer in model.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]

	
	# 1 extra list for the epoch
	fibers_vs_time = {(i, str(layer)):[] for i, layer in enumerate(layers)}
	fibers_vs_time['epoch'] = []

	epoch_zero = has_epoch_zero
	for pt_file in pt_files:
		model = torch.load(pt_file, weights_only=False, map_location='cpu')
		epoch = int(pt_file.split('/')[-1].split('.')[0].split('_')[1]) + 1
		
		if epoch_zero:
			reinitialize_weights(model)
			epoch_zero = False
			epoch = 0
			
		if hasattr(model.policy, 'policy'):
			layers = [layer for layer in model.policy.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))] + [model.policy.recurrent]
		else:
			layers = [layer for layer in model.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]

		clusters = None
		prev_layer = None
	
		for idx, layer in enumerate(layers):
			if isinstance(layer, nn.Conv2d):
				clusters = fibration_conv2d(weights = layer.weight.data, 
												in_clusters = clusters, 
												threshold = threshold, 
												first_layer = True if idx == 0 else False,
												bias = layer.bias.data)
				fibers_vs_time[(idx, str(layer))].append(clusters)
				prev_layer = layer
				
			elif isinstance(layer, nn.Linear):
				if isinstance(prev_layer, nn.Conv2d):
					clusters = [x for x in clusters for _ in range(layer.in_features // prev_layer.out_channels)]
					
				clusters = fibration_linear(weights = layer.weight.data, 
												in_clusters = clusters, 
												threshold = threshold, 
												first_layer = True if idx == 0 else False,
												bias = layer.bias.data)
				fibers_vs_time[(idx, str(layer))].append(clusters)
				prev_layer = layer
			elif isinstance(layer, nn.LSTM):
				clusters = fibration_lstm(layer, clusters, threshold)
				fibers_vs_time[(idx, str(layer))].append(clusters)
				prev_layer = layer

		fibers_vs_time['epoch'].append(epoch)
		
	fibers_vs_time = sort_dict_by_key(fibers_vs_time, 'epoch')
	
	return fibers_vs_time


def get_opfibers_vs_time(pt_files, threshold, has_epoch_zero=False):
	
	model = torch.load(pt_files[0], weights_only=False, map_location='cpu')
	
	if hasattr(model.policy, 'policy'):
		layers = [layer for layer in model.policy.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))] + [model.policy.recurrent]
	else:
		layers = [layer for layer in model.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))]

	# 1 extra list for the epoch
	opfibers_vs_time = {(i, str(layer)):[] for i, layer in enumerate(layers)}
	opfibers_vs_time['epoch'] = []

	epoch_zero = has_epoch_zero
	for pt_file in pt_files:
		model = torch.load(pt_file, weights_only=False, map_location='cpu')
		epoch = int(pt_file.split('/')[-1].split('.')[0].split('_')[1]) + 1
		
		if epoch_zero:
			reinitialize_weights(model)
			epoch_zero = False
			epoch = 0
			
		if hasattr(model.policy, 'policy'):
			layers = [model.policy.policy.actor] + [model.policy.recurrent] + [layer for layer in model.policy.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))][::-1]
		else:
			layers = [model.policy.actor] + [layer for layer in model.policy.network.children() if isinstance(layer, (nn.Conv2d, nn.Linear))][::-1]

		clusters = None
		prev_layer = None
	
		for idx, layer in enumerate(layers):
			if isinstance(layer, nn.Conv2d):
				clusters = opfibration_conv2d(weights = layer.weight.data, 
												in_clusters = clusters, 
												threshold = threshold, 
												first_layer = True if idx == 0 else False,
												bias = layer.bias.data)
				opfibers_vs_time[(idx, str(layer))].append(clusters)
				prev_layer = layer
				
			elif isinstance(layer, nn.Linear):
				if isinstance(prev_layer, nn.Conv2d):
					clusters = [x for x in clusters for _ in range(layer.in_features // prev_layer.out_channels)]
					
				clusters = opfibration_linear(weights = layer.weight.data, 
												out_clusters = clusters, 
												threshold = threshold, 
												last_layer = True if idx == 0 else False,
												bias = layer.bias.data)
				opfibers_vs_time[(idx, str(layer))].append(clusters)
				prev_layer = layer
			elif isinstance(layer, nn.LSTM):
				clusters = opfibration_lstm(layer, clusters, threshold)
				opfibers_vs_time[(idx, str(layer))].append(clusters)
				prev_layer = layer

		opfibers_vs_time['epoch'].append(epoch)
		
	opfibers_vs_time = sort_dict_by_key(opfibers_vs_time, 'epoch')
	
	return opfibers_vs_time