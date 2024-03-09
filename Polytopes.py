
import copy
import numpy as np
from scipy.spatial import Delaunay
import cdd 
import pickle
from scipy.linalg import null_space, orth
from scipy.spatial import ConvexHull


def get_indices_of_array(base_list, test_array):
    '''
    helper function 
    '''
    index_list = []
    for index in range(len(base_list)):
        if np.array_equal(base_list[index], test_array):
            index_list.append(index)
    return index_list

def max_diff(array_list):
    '''
    helper function
    '''
    dist_matrix = np.zeros((len(array_list), len(array_list)))
    for i in range(len(array_list)):
        for j in range(len(array_list)):
            dist_matrix[i][j] = np.linalg.norm(array_list[i]-array_list[j])
    return np.max(dist_matrix)      
    

def get_colorschemes(k, dim):
    '''
    helper function for edgewise subdivision 
    
    generates a list of so-called "color schemes" in matrix form, each of 
    which will correspond to a simplex in the edgewise subdivision later.
    
    Parameters:    
        k: integer
            The parameter of the desired edgewise subdivision
        dim: integer
            The dimension (number of vertices - 1) of the simplex to be 
            subdivided
    
    Returns:
        lst_of_matrices: list of numpy arrays
            The desired list of all possible color schemes for parameter k 
            and dimension dim.
    '''
    lst=[([0],[])]
    while len(lst[0][0])<dim*k:
        templst=[]
        for (scheme,jumps) in lst:
            templst.append((scheme+[scheme[-1]],jumps))
            if scheme[-1] != dim-1:
                if len(scheme)%dim != 0:
                    templst.append((scheme+[scheme[-1]+1],jumps+[len(scheme)%dim]))
                else:
                    templst.append((scheme+[scheme[-1]+1],jumps))
        lst=templst
    templst=copy.deepcopy(lst)
    for (scheme,jumps) in lst:
        if len(set(jumps))!=dim-1:
            templst.remove((scheme,jumps))
    lst=templst
    lst_of_matrices=[]
    for (scheme,jumps) in lst:
        matrix=[]
        for row in range(0,k):
            temp=[]
            for column in range(0,dim):
                temp.append(scheme[row*dim+column])
            matrix.append(temp)
        matrix=np.array(matrix)
        lst_of_matrices.append(matrix)
    return lst_of_matrices


def colorscheme_to_simplex(matrix,simplex,index_to_pt_list,known_combs):
    '''
    helper function for edgewise subdivision 
    
    converts a color scheme generated with get_colorschemes into the new 
    simplex of the edgewise subdivision as an instance of the class Polytope.
    
    Parameters:
        matrix: numpy array
            The color scheme in matrix form, which is now converted to a 
            simplex
        simplex: instance of class Polytope
            The simplex to be subdivided
        index_to_pt_list: list of numpy arrays
            The list containing all points (and possibly more) the simplex
            references with the indexes given in self.points.
        known_combs: list of lists
            A list of previously added points (in their baricentric coordinates
            relative to the simplex) which have previously been added to the 
            index_to_pt_list and therefore do not have to be added again
            
    Returns:
        simplex: Instance of class Polytope
            The resulting simplex generated from the given colorscheme in 
            'matix'
        
    '''
    dimension=len(index_to_pt_list[0])
    aktuelle_itpl=[index_to_pt_list[i] for i in simplex.points]
    points=[]
    conns=[]
    for i in range(0,len(matrix[0])):
        column=matrix[:,i]
        if tuple(column) in known_combs.keys():
            point=known_combs[tuple(column)]
            points.append(point)
        else:
            point=np.zeros(dimension)
            for x in column:
                point+=aktuelle_itpl[x]
            point=point*(1/len(column))    
            index_to_pt_list.append(point)
            points.append(len(index_to_pt_list)-1)
            known_combs[tuple(column)]=len(index_to_pt_list)-1
    for j in range(0,len(points)):
        for l in range(0,len(points)):
            if j>l:
                conns.append((l,j))
    simplex=Polytope(points,conns,simplex.aff_lin)
    return simplex

def prep_for_eval_on_a_point(LinearSegment):
    '''
    prepare a given polytope with an associated affine linear map 
    for application of PLT by calculating necessary information 
    '''
    W, b = LinearSegment.aff_lin
    N = null_space(W)
    R = orth(W)
    if N.size == 0:
        if R.shape[0] == R.shape[1]:
            #bijective case
            W_inv = np.linalg.inv(W)
            det_W_inv = np.abs(np.linalg.det(W_inv))
            return (0, W_inv, det_W_inv)
        else:
            #injective, but not surjective case
            RT = np.transpose(R)
            W_tilde = RT@W
            b_tilde = RT@b
            W_tilde_inv = np.linalg.inv(W_tilde)
            det_W_tilde_inv = np.abs(np.linalg.det(W_tilde_inv))
            return (1, W_tilde_inv, det_W_tilde_inv, RT, b_tilde)

    else:
        if R.shape[1] == W.shape[0]:
            #surjective, but not injective case
            W_pi = np.linalg.pinv(W)
            N_orth = orth(W_pi)
            
            W_tilde = W@N_orth
            W_tilde_inv = np.linalg.inv(W_tilde)
            det_W_tilde_inv = np.abs(np.linalg.det(W_tilde_inv))
            
            return (2, det_W_tilde_inv)
        else:
            #not injective and not surjective case 
            RT = np.transpose(R)
            
            W_pi = np.linalg.pinv(W)
            N_orth = orth(W_pi)
            
            W_tilde = RT@W@N_orth
            W_tilde_inv = np.linalg.inv(W_tilde)
            det_W_tilde_inv = np.abs(np.linalg.det(W_tilde_inv))
            return (3, det_W_tilde_inv) 


class Polytope():
    '''
    Class for a single polytope as part of a convex polytope union
    '''
    def __init__(self, points, conn, aff_lin):
        '''
        create a polytope with an affine linear map saved along with it
        
        Parameters:
            points: list of integers
                List of indices of vertices of CPU which are vertices of the polytope
            conn: list
                List of pairs of vertex indices which are connected by an edge
                All vertex indices appearing in conn must appear in points as well
            aff_lin: tuple of length 2
                aff_lin[0]: 2D numpy array, matrix describing affine linear map
                aff_lin[1]: 1D numpy array, bias vector of affine linear map 
        '''
        self.points = list(points)
        self.conn = conn
        self.aff_lin = aff_lin
        
        #Inequalities for H-representations of the polytope in input space, initialized with None
        self.equations = None
        #Inequalities for H-representations of the polytope in output space, initialized with None 
        self.output_equations = None 
    
    def delaunay_triang(self, index_to_pt_list):
        '''
        Delaunay triangulation of polytope 
        '''
        simps = []
        pt_list = [index_to_pt_list[i] for i in self.points]
        if len(index_to_pt_list[0]) > 1:
            tri = Delaunay(pt_list)        
            for sim in tri.simplices:
                sim_pts = [self.points[j] for j in sim]
                sim_conns = []
                for i in sim_pts:
                    for j in sim_pts:
                        if i>j:
                            sim_conns.append((j,i))
                simps.append(Polytope(sim_pts, sim_conns, self.aff_lin))
            return simps
        else:
            return [self]

        
    def edgewise(self, index_to_pt_list, k = 2, known_colorschemes={}):     
       '''
       Edgewise subdivision of polytope 
       
       Parameters:
           index_to_pt_list: list of numpy arrays
               The list containing all points (and possibly more) the polytope
               references with the indexes given in self.points
           k: integer
               The parameter of the edgewise subdivision. Every edge of the 
               polytope will be divided into k edges of equal length.
           known_colorschemes: dictionary 
               The keys are tuples (dim,k) and it saves all colorschemes for a
               simplex of dimension dim and parameter k, so that they do not
               have to be calculated again
        
        Returns:
            simplices: list of instances of the class polytope
               The resulting edgewise subdivision
       
       '''
       dim=len(self.points)
       known_combs1={}
       
       if (dim,k) in known_colorschemes.keys():
           colorschemes=known_colorschemes[(dim,k)]
       else:
           colorschemes=get_colorschemes(k, dim)
           known_colorschemes[(dim,k)]=colorschemes
       simplices=[]
       for matrix in colorschemes:
           simplices.append(colorscheme_to_simplex(matrix, self, index_to_pt_list, known_combs1))
       return simplices       
            
   
class CPU():
    '''
    Class representing a union of convex polytopes 
    '''

    def __init__(self, input_points, conn):
        '''
        initializes an instance of class CPU consisting of 1 polytope
        defined by the given points and connections
        
        Parameters:
            input_points: list
                Vertices of the polytope
            conn: list
                Pairs of vertices which are connected by an edge   
        '''
        #self.input_points_union contains points in the input space 
        self.input_points_union = copy.deepcopy(input_points)
        
        #initialize affine linear maps
        #each polytope in the CPU is initialized with aff_lin as its linear map 
        aff_lin = (np.identity(input_points[0].size),
                   np.zeros(input_points[0].size))
        
        #list of all polytopes in the CPU 
        self.polytopes = list(set([Polytope(set(range(len(input_points))), conn, aff_lin)]))
        
        #self.current_points_union contains the points after propagation through linear and piecewise linear functions
        self.current_points_union = copy.deepcopy(input_points)
        
        #list of all pairs of CPU vertices connected by an edge 
        self.conn = conn
        
        #possible subdivision of CPU by Delaunay triangulation and edgewise subdivision 
        self.subdivision = None
        
        #possible additional information about affine linear maps for distribution propagation 
        self.lin_seg_preps = None
    
    def linear_transform(self, weights = None, bias = None):
        '''
        transform CPU defined by self.current_points_union 
        by propagation through an affine linear map 
        
        Parameters:
            weights: 2D numpy array
                The matrix describing the linear map
                If weights is None, the identity matrix is used
                Default is None
            bias: 1D numpy array 
                The bias vector of the affine linear map 
                If bias is None, the zero vector is used
                Default is None 
        '''
        if weights is None:
            weights = np.identity(self.point_dim)
            
        if bias is None:
            bias = np.zeros(weights.shape[1])
            
        for i, point in enumerate(self.current_points_union):
            self.current_points_union[i] = np.matmul(weights, point) + bias
        
        for P in self.polytopes:
            P.aff_lin = (
                weights @ P.aff_lin[0], weights @ P.aff_lin[1] + bias
                )
            
        
    def divide_by_activation(self, dimension, alpha = 0.01):
        '''
        divide the each polytope in the CPU spanned by self.input_points_union 
        according to LeakyReLu activation on self.current_points_union in the
        given dimension
        
        Parameters:
            dimension: integer
                The dimension of self.current_points_union to which the
                activation function is applied
                Enumeration of dimension starts at 1
            alpha: float
                The parameter of the LeakyReLu activation function
                Default is 0.01
        '''
        
        dim = dimension -1
        
        point_upper_lower_list = [0] * len(self.current_points_union)
        
        #check which points lie in the upper resp. lower halfspace
        for k, point in enumerate(self.current_points_union):
            if point[dim] > 0:
                point_upper_lower_list[k] = 1
            elif point[dim] < 0:
                point_upper_lower_list[k] = -1
            else:
                point_upper_lower_list[k] = 0
        
        #Set of all edges intersecting the separating hyperplane given by activation function 
        conn_intersec = set(
            (a,b) for (a,b) in self.conn if 
            np.abs(point_upper_lower_list[a]-point_upper_lower_list[b]) == 2
            )
        
        conn_decomp = dict.fromkeys(conn_intersec)
        for (a,b) in conn_intersec:
            #calculation of intersection of edge and hyperplane 
            curr_point1 = self.current_points_union[a]
            curr_point2 = self.current_points_union[b]
            curr_direc = curr_point2 - curr_point1
            
            input_point1 = self.input_points_union[a]
            input_point2 = self.input_points_union[b]
            input_direc = input_point2 - input_point1
            
            s = -curr_point1[dim]/curr_direc[dim]
            input_intersec = input_point1 + s * input_direc
            curr_intersec = curr_point1 + s * curr_direc
            
            idx = get_indices_of_array(self.input_points_union, input_intersec)
            if idx == []:
                self.input_points_union.append(input_intersec)
                self.current_points_union.append(curr_intersec)
               
                
                input_intersec_idx = len(self.input_points_union)-1
                
                #update point_upper_lower_list
                point_upper_lower_list.append(0)
                
            else:
                input_intersec_idx = idx[0]
                if point_upper_lower_list[input_intersec_idx] != 0:
                    print('Compatibility problem with point_upper_lower_list')
            
            #create necessary new edges after subdivision 
            if point_upper_lower_list[a] < point_upper_lower_list[b]:
                conn_lower = order((a, input_intersec_idx))
                conn_upper = order((input_intersec_idx, b))
            else:
                conn_lower = order((b, input_intersec_idx))
                conn_upper = order((input_intersec_idx, a))
            
           
            conn_decomp[(a,b)]= (conn_lower, conn_upper, input_intersec_idx)
        
        #update polytope list of CPU 
        new_polytopes = set()
        for P in self.polytopes:
            P.points = set(P.points)
            P_lower_points = set(k for k in P.points if point_upper_lower_list[k] <= 0)
            P_upper_points = set(k for k in P.points if point_upper_lower_list[k] >= 0)
            
            #case where polytope P lies in upper halfspace
            if P_upper_points ==  P.points:
                new_polytopes.add(P)
                
            #case where polytope P lies in lower halfspace 
            elif P_lower_points == P.points:
                #update aff_lin of P 
                shape = P.aff_lin[1].shape
                x = np.ones(shape)
                x[dim] = alpha
                P.aff_lin = (np.diag(x)@P.aff_lin[0], np.diag(x)@P.aff_lin[1])
                
                new_polytopes.add(P)
            
            #case where polytope P has non-empty intersection with both halfspaces 
            else:
              
                #intersection of P and hyperplane 
             
                P_new_points = {conn_decomp[(a,b)][2] for (a,b) in P.conn 
                                if (a,b) in conn_intersec}
                
             
                P_lower_points.update(P_new_points)
                
                #create non-reduced set of edges 
                P_lower_conn = set((a,b) for (a,b) in P.conn if point_upper_lower_list[a]<=0
                                   and point_upper_lower_list[b]<=0)
                
                P_lower_conn.update([conn_decomp[(a,b)][0] for (a,b) in P.conn
                                    if (a,b) in conn_intersec])
                
                P_points_on_hp = P_new_points.union(
                    {k for k in P.points if point_upper_lower_list[k] == 0}
                    )
                
                P_new_conn = {(a,b) for a in P_points_on_hp for b in
                              P_points_on_hp if a < b}
                
               
                P_lower_conn.update(P_new_conn)
                
             
                #reduce set of edges of P 
                P_lower_points_reduced, P_lower_conn_reduced = reduce(
                    self.input_points_union, P_lower_points, P_lower_conn)
                
               
                
                #update aff_lin of P 
                shape = P.aff_lin[1].shape
                x = np.ones(shape)
                x[dim] = alpha
                P_lower_aff_lin = (np.diag(x)@P.aff_lin[0], np.diag(x)@P.aff_lin[1])
               
                #add the intersection of P and lower halfspace to polytope list of CPU
                new_polytopes.add(Polytope(P_lower_points_reduced,
                                          P_lower_conn_reduced,
                                          P_lower_aff_lin))
                
                #create intersection of P and upper halfspace is an analogue way
                P_upper_points.update(P_new_points)
                P_upper_conn = set((a,b) for (a,b) in P.conn if point_upper_lower_list[a]>=0
                                   and point_upper_lower_list[b]>=0)
                P_upper_conn.update([conn_decomp[(a,b)][1] for (a,b) in P.conn if
                                 (a,b) in conn_intersec])
                P_upper_conn.update(P_new_conn)
                
                P_upper_points_reduced, P_upper_conn_reduced = reduce(
                   self.input_points_union, P_upper_points, P_upper_conn)
                
               
                #add the intersection of P and upper halfspace to polytope list of CPU 
                new_polytopes.add(Polytope(P_upper_points_reduced,
                                          P_upper_conn_reduced,
                                         P.aff_lin))
            P.points = list(P.points)
        self.polytopes = list(new_polytopes)
        
        
        #update self.current_points_union
        for k in range(len(self.current_points_union)):
            if point_upper_lower_list[k] == -1:
                self.current_points_union[k][dim] = alpha * self.current_points_union[k][dim]
        
        #update self.conn 
        self.conn = set().union(*[P.conn for P in self.polytopes])


    def getInputEquations(self):
        '''
        Get H-representation of all polytopes of CPU in the input space and
        adds them as attribute equations to each polytope
        Remove lower-dimensional polytopes from polytope list of CPU  
        '''
        
        del_indices = []
        for i in range(0, len(self.polytopes)):
            polytope_points=[self.input_points_union[j] for j in self.polytopes[i].points]
            if len(self.input_points_union[0]) >=2:
                try:
                    polytope_equations = ConvexHull(polytope_points).equations
                    self.polytopes[i].equations = polytope_equations 
                except:
                    del_indices.append(i)
            else:
                polytope_equations = []
                for pt in polytope_points:
                    polytope_equations.append(np.array([-1, pt[0]]))
                self.polytopes[i].equations = np.array(polytope_equations)
               
        
        self.polytopes = [self.polytopes[i] for i in range(0, len(self.polytopes)) if i not in del_indices]
        
        return 


    def getOutputEquations(self):
        '''
        calculates the H-representations for all polytopes in CPU in the output space
        and adds them as attribute output_equations in the polytope list
        
        '''
        
        for i in range(0, len(self.polytopes)):
            polytope_points = [self.current_points_union[i] for i in self.polytopes[i].points]
            try:
                polytope_equations = ConvexHull(polytope_points).equations
                self.polytopes[i].output_equations = polytope_equations
            except:
                
                polytope_vertices = [np.append(np.ones(1), self.current_points_union[j]) for j in self.polytopes[i].points]
                mat = cdd.Matrix(polytope_vertices, number_type='float')
                mat.rep_type = cdd.RepType.GENERATOR
                poly = cdd.Polyhedron(mat)
                inequalities = poly.get_inequalities()
                equality_indices = inequalities.lin_set
                
                a = np.array(inequalities)
                b = -np.append(a[:,1:], np.reshape(a[:,0], (a.shape[0],1)), axis=1)
                for index in equality_indices:
                    columns = b.shape[1]
                    b = np.append(b, np.reshape(-b[index,:], (1,columns)), axis=0)
                
                self.polytopes[i].output_equations = b 
                    
        return 


    def checkPolytopes(self, points, eps=0):
        '''
        check which output polytopes each point in a list of points in contained in
        
        Parameters:
            points: list of numpy arrays
                list of points in the output space
            eps: float
                tolerance for fulfilling output equations
                For each inequality in the H-representation of a polytope,
                it is checked if the point satisfies it up to a tolerance
                eps
                Default is 0
        
        Returns:
            points_polytope_indices: list of lists
                List containing a list of polytopes indices containing this point
                for each point in parameter points 
        '''
      
        points_polytope_indices = []
        for point in points:
            point_polytope_indices = []
            for i in range(0, len(self.polytopes)):
                if self.polytopes[i].output_equations is not None:
                    
                    eq = self.polytopes[i].output_equations
                      
                    for j in range(0, len(eq)):
                        if np.dot(point, eq[j][0:-1]) > -eq[j][-1]+eps:
                            #if one equation is not fulfilled up to eps, point is not containted in polytope 
                            break
                        #if all equations have been tested and satisfied, add polytope i to list
                        if j==len(eq)-1:
                            point_polytope_indices.append(i)
                
                else:
                    polytope_vertices = [ np.append(np.ones(1), self.current_points_union[j]) for j in self.polytopes[i].points]
                    if cdd_in_hull(np.array(polytope_vertices), point, eps):
                        point_polytope_indices.append(i)
    
            points_polytope_indices.append(point_polytope_indices)
                    
        return points_polytope_indices
    
    def saveCPU(self, filename= "cpu.pkl"):
        '''
        save an instance of class CPU as a pickle file
        
        Parameters:
            filename: String
                Name of the pickle file 
        '''
        
        cpu_dict={}
        cpu_dict['current_points_union'] = self.current_points_union
        cpu_dict['input_points_union'] = self.input_points_union
        cpu_dict['subdivision'] = self.subdivision
        cpu_dict['lin_seg_preps'] = self.lin_seg_preps
        for i in range(0, len(self.polytopes)):
            cpu_dict[i] = {'points': self.polytopes[i].points,
                           'conn': self.polytopes[i].conn,
                           'aff_lin': self.polytopes[i].aff_lin,
                           'equations': self.polytopes[i].equations,
                           'output_equations': self.polytopes[i].output_equations}
        
        with open(filename, 'wb') as f:
            pickle.dump(cpu_dict, f)
        
        return 
    
    def prep_lin_seg_for_eval(self):
        list_of_preparations=[]
        for segment in self.polytopes:
            list_of_preparations.append(prep_for_eval_on_a_point(segment))
        self.lin_seg_preps=list_of_preparations

                
    def subdivide(self, k = 2, epsilon = None, mode = "delaunay"):
        '''
        apply Delaunay triangulation to all polytopes in the CPU, followed by 
        an edewise subdivision of each of the resulting simplices.
        
        Parameters:
            k: integer
                The parameter of the edgewise subdivision. Every edge of the 
                polytope will be divided into k edges of equal length.
            epsilon: float
                If provided, this yields the threshold that the length of every 
                edge of a resulting polytope has fall under. The edgewise
                subdivison is scaled accordingly
            mode: string
                Only mode implemented right now: "delaunay". Determines the 
                method, how the original polytope should be divided into
                simplices
        '''
        edgewise_itpl = copy.deepcopy(self.input_points_union)
        subsimplices = []
        known_colorschemes_1={}
        tri_list = []
        for i, poly in enumerate(self.polytopes):
            bary_list = []
            if i%10 == 0:
                print("Polytope number " + str(i))
                
            
            if mode == "delaunay":
                bary = poly.delaunay_triang(edgewise_itpl)
                bary_only_points=[x.points for x in bary]
                tri_list.append(bary_only_points)
                
            if epsilon is None:
                for bar in bary:
                    edge = bar.edgewise(edgewise_itpl, k, known_colorschemes=known_colorschemes_1)
                    edge_only_points = [x.points for x in edge]
                    bary_list.append(edge_only_points)
            else:
                k_ad_list = []
                for bar in bary:
                    points_in_bar=[edgewise_itpl[j] for j in bar.points]
                    widest_edge = max_diff(points_in_bar)
                    k_ad = int(np.ceil(widest_edge/epsilon))
                    k_ad_list.append(k_ad)
                    edge = bar.edgewise(edgewise_itpl, k_ad, known_colorschemes=known_colorschemes_1)
                    edge_only_points=[x.points for x in edge]
                    bary_list+=edge_only_points
            subsimplices.append(bary_list)
            
        self.subdivision = tri_list, subsimplices, edgewise_itpl     
            

def loadCPU(filename):
    '''
    load a CPU from a pickle file containing a saved CPU
    
    
    Parameters:
        filename: String
            the name of the file containing the CPU 
    
    Returns:
        cpu: instance of class CPU
            The loaded CPU
            If file does not exist, an empty CPU without points, 
            conns and polytopes is returned 
    '''
        
    try:
        with open(filename, 'rb') as f:
            cpu_dict = pickle.load(f)
    except:
        print("File does not exist!")
        return CPU([], [])
        
    cpu = CPU(cpu_dict['input_points_union'], [])
    cpu.current_points_union = cpu_dict['current_points_union']
    try:
        cpu.lin_seg_preps=cpu_dict['lin_seg_preps']
    except:
        cpu.lin_seg_preps = None
    try:
        cpu.subdivision = cpu_dict['subdivision']
    except:
        cpu.subdivision = None
    cpu.polytopes = []
    polytope_number_list = [key for key in cpu_dict.keys() if type(key)==int]
    polytope_number_list.sort()
    for i in polytope_number_list:
        p = Polytope(cpu_dict[i]['points'], cpu_dict[i]['conn'], cpu_dict[i]['aff_lin'])
        p.equations = cpu_dict[i]['equations']
        p.output_equations = cpu_dict[i]['output_equations']
        cpu.polytopes.append(p)
    
    return cpu 
            

def cdd_in_hull(points, x, eps=10**(-10)):
    '''
    check if a point x is contained in the convex hull of a list of points 
    
    Parameters:
        points: list of numpy arrays
            List of points spanning the convex hull
        x: numpy array
            Point to check for
        eps: float
            Tolerance up to which the inequalities in the H-representation of
            the convex hull must be satisfied by x
            Default is 10^(-6)
    
    Returns:
        True, if x lies in convex hull
        False, if x does not lie in convex hull 
    '''
  
    mat = cdd.Matrix(points, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    
    inequalities = poly.get_inequalities()
    equality_indices = inequalities.lin_set
    
    a = np.array(inequalities)
   
    rows = a.shape[0]
    for i in range(0, rows):
        if i not in equality_indices:
            if -np.dot(a[i,1:].flatten(), x) > a[i,0] + eps:
                return False 
    
        else:
            if abs(-np.matmul(a[i,1:], x) - a[i,0]) > eps:
                return False 
        
    return True 


def reduce(pts_list, pts_idx, conn):
    '''
    helper function for divide_by_activation: reduce the list of vertices and edges
    of the intersection of a polytope, if possible 
    '''
    try:
        pts_idx_list = list(pts_idx)
        pts = [pts_list[k] for k in pts_idx]
        hull = ConvexHull(pts)
        pts_reduced =  [l for (k,l) in enumerate(pts_idx_list) if k in hull.vertices]
        
        conn_reduced_list = [None] * len(hull.simplices)
        for k,s in enumerate(hull.simplices):
            
            s_idx_list = [pts_idx_list[k] for k in s]
            conn_reduced_list[k] = set((a,b) for (a,b) in conn if a in 
                                       s_idx_list and b in s_idx_list)
        conn_reduced = set().union(*conn_reduced_list)
        return set(pts_reduced), conn_reduced
    
    except:
        return pts_idx, conn

def order(P):
    '''
    order a tuple of 2 points 
    '''
    if P[0] <= P[1]:
        return P
    else:
        return (P[1], P[0])
