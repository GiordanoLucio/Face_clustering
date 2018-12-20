######################################################################
#	Matching function												 #
#																	 #
#	Use:															 #
#	matching = calculate_matching(flow)								 #
#																	 #
# This function takes as an input a nxn matrix in which each element #
# flow[i][j] contains the probability that subject i is associated 	 #
# with cluster j and n is the number of subjects (= #clusters).		 #
# The function returns an array, called matching, in which each 	 #
# element matching[k] contains the generated label of the cluster 	 #
# corresponding to the subject originally labelled k.				 #
# We used PuLP library to solve the matching problem, as we 		 #
# formulated it as a linear programming problem. 					 #
# We have a bipartite graph in which the first group of nodes is 	 #
# used to represent subjects and the second group to represent 		 #
# clusters. Each edge going from subject node i to cluster node j 	 #
# represents the fact that they are corresponding and it's got a 	 #
# weight, which is flow[i][j].										 #
# We have to choose the right edges to take as good assignments,	 #
# maximizing the overall probability.								 #
# We formulated the problem as the following:						 #
# "Minimize summation of -flow[i][j]*x[i][j] for each i, for each j" #
# where i and j go from 0 to n-1, respecting these constraints:		 #
# 1) Exactly n variables must assume value of 1, the rest must be 0. #
# The non-zeroes variables will represent the correct matching 	 	 #
# between subjects and clusters.									 #
# 2) Exactly one cluster must be associated with each subject.		 #
# 3) Exactly one subject must be associated with each cluster.		 #
# PuLP solver uses Branch and Cut algorithm, which is an 			 #
# optimization of the classic simplex method, specialized for 		 #
# integer linear programming problems.								 #
# After solving the problem, we build the solution array assigning	 #
# to the i-th cell the value of the cluster label associated to the  #
# i-th subject. 													 #
# We can modify the code just changing a single line and we can get  #
# a solution array in which we have the subject associated to the 	 #
# i-th cluster in the i-th position. 								 #
######################################################################

def calculate_matching(flow):
  #number of subjects = number of clusters (one-to-one relationship)
  n = flow[0].size 
  
  #initialize problem
  problem = LpProblem("From Label to Cluster", LpMinimize)
  
  #GENERATE LP PROBLEM VARIABLES
  x = [[]]*n
  for i in range(0, n):
    x[i] = np.empty(n, dtype=LpVariable)
    for j in range(0, n):
      #generate variable which represents flux on edge (u, v) for all u representing labels and for all v representing clusters
      exec("x[%d][%d] = LpVariable(\"x_%d_%d\",0,1,LpInteger)" % (i, j, i, j))      
  
  #SET THE OBJECTIVE FUNCTION
  problem += sum((-flow[i][j])*x[i][j] for i in range(0, n) for j in range(0, n)), "Objective function"
  
  #COSTRAINT (1)
  problem += sum(x[i][j] for i in range(0, n) for j in range(0, n)) == n, "Exactly n couples found"
  
  #COSTRAINTS SET (2)
  for i in range(0, n):
    str = "Exactly one cluster for label %d" %i
    problem += sum(x[i][j] for j in range(0, n)) == 1, str
    
  #COSTRAINTS SET (3)
  for j in range(0, n):
    str = "Exactly one label for cluster %d" %j
    problem += sum(x[i][j] for i in range(0, n)) == 1, str
    
  #SOLVE
  matching = [0]*n
  if problem.solve() == 1: #OPTIMUM OBTAINED: SET RETURN VALUES
    for v in problem.variables():
      if v.varValue == 1: #edge (i,j) selected as an assignment
        split = v.name.split("_")
        i = int(split[1])
        j = int(split[2])
        #uncomment to return an array where the i-th value is our custom label corresponding to the i-th cluster
        #matching[j] = i
        #uncomment to return an array where the i-th value is the cluster corresponding to the i-th subject
        matching[i] = j
        
  else: 
    print("Matching not found!")
    
  return matching