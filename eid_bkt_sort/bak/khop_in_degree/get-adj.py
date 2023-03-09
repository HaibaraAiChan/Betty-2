import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'../../../pytorch/utils/')
sys.path.insert(0,'../../../pytorch/micro_batch_train/')
sys.path.insert(0,'../../../pytorch/models/')

import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random

import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset, load_ogb
from memory_usage import see_memory_usage, nvidia_smi_usage

from cpu_mem_usage import get_memory
from statistics import mean



import pickle
from utils import Logger
import os 
import numpy
from ortools.sat.python import cp_model as cp
import math, sys
from cp_sat_utils import scalar_product
from scipy import sparse

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)


#### Entry point
def run(args, device, data):
	
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])

	model = cp.CpModel()
	print('the adj of the raw graph')
	# print(g.adj_sparse('coo')[0].tolist())
	# print(g.adj_sparse('coo')[1].tolist())
	# print(g.adj(scipy_fmt='coo').todense())
	# print()
	print(len(g.adj_sparse('coo')[0].tolist()))
	print(len(g.adj_sparse('coo')[1].tolist()))
	# print(g.adj(scipy_fmt='coo').todense())
	print()
	lil_matrix = g.adj(scipy_fmt='coo').tolil()
	print('lil_matrix.rows', lil_matrix.rows[:10])

	
	# karate
	# n_subset = 7
	# n_nid = 6

	# cora
	n_subset = 2708
	n_nid = 2708

	SUBSET_ID = list(range(n_subset))
	NID = list(range(n_nid))
	Cost = np.ones(n_subset, dtype=int)
	print('Cost ', Cost)
	transposeAdj = lil_matrix.rows
	print('transposeAdj ', transposeAdj)

	#
	# variables
	#
	Use_subset = [model.NewBoolVar("Use_subset[%i]" % w) for w in SUBSET_ID]
	total_cost = model.NewIntVar(0, n_subset * sum(Cost), "total_cost")

	#
	# constraints
	#
	scalar_product(model, Use_subset, Cost, total_cost)

	for j in NID: # NID : dst nodes
		# Sum the cost for use the subsets 

		# print('j',j)
		tmp = [Use_subset[c] for c in transposeAdj[j]]
		# print(tmp)
		print()
		print("row "+str(j)+ ' -----'+ str(sum(tmp)))

		model.Add(sum([Use_subset[c ] for c in transposeAdj[j]]) >= 1)

	# objective: Minimize total cost
	model.Minimize(total_cost)

	#
	# search and result
	#
	solver = cp.CpSolver()
	status = solver.Solve(model)
	
	if status == cp.OPTIMAL:
		print("Total cost", solver.Value(total_cost))
		print("We should use these subsets: ", end=" ")
		for w in SUBSET_ID:
			if solver.Value(Use_subset[w]) == 1:
				print(w, end=" ")
		print()

	print()
	print("NumConflicts:", solver.NumConflicts())
	print("NumBranches:", solver.NumBranches())
	print("WallTime:", solver.WallTime())



	return



def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=1,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--aggre', type=str, default='mean')
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	# argparser.add_argument('--selection-method', type=str, default='metis')
	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=1)

	argparser.add_argument('--re-partition-method', type=str, default='REG')
	# argparser.add_argument('--re-partition-method', type=str, default='random')
	argparser.add_argument('--num-re-partition', type=int, default=0)

	# argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=5)

	argparser.add_argument('--num-hidden', type=int, default=6)

	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')
	argparser.add_argument('--fan-out', type=str, default='10')
	# argparser.add_argument('--num-layers', type=int, default=2)
	# argparser.add_argument('--fan-out', type=str, default='2,4')
	# argparser.add_argument('--fan-out', type=str, default='10,25')
	
	argparser.add_argument('--log-indent', type=float, default=2)
#--------------------------------------------------------------------------------------
	

	argparser.add_argument('--lr', type=float, default=1e-3)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

