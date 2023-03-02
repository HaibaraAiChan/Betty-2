
# from __future__ import print_function
from ortools.sat.python import cp_model as cp
import math, sys
from cp_sat_utils import scalar_product
from scipy import sparse
import numpy as np

def main():

  model = cp.CpModel()
  #
  # data
  #
  
  x = 4  # number of subsets
  X = list(range(x))
  y = 6  # number of atoms
  Y = list(range(y))
  Adj = [[2,3],[2],[0,1,4],[0,2,5]]   # subset row, atom col
  mtx = sparse.lil_matrix((x, y)) # matrix : 4*6
  for i in range (len(Adj)):
    mtx[i,Adj[i]] = np.ones(len(Adj[i]))
  # print(mtx)
  # print(mtx.todense())
  T_mtx= mtx.transpose()
  print(T_mtx.todense()) 
  print(T_mtx.rows)
  transposeAdj = T_mtx.rows
  # transposeAdj = [[2,3],[2],[0,1,3],[0],[2],[3]] # atom row, subset col
  
  Cost = [1, 1, 1, 1] # each subsets' weight
  #
  # variables
  #
  Use_subset = [model.NewBoolVar("Use_subset[%i]" % w) for w in X]
  total_cost = model.NewIntVar(0, x * sum(Cost), "total_cost")

  #
  # constraints
  #
  scalar_product(model, Use_subset, Cost, total_cost)

  for j in Y:
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
    for w in X:
      if solver.Value(Use_subset[w]) == 1:
        print(w, end=" ")
    print()

  print()
  print("NumConflicts:", solver.NumConflicts())
  print("NumBranches:", solver.NumBranches())
  print("WallTime:", solver.WallTime())


if __name__ == "__main__":
  main()

