
import torch
import numpy as np
import copy
from itertools import chain, combinations
import re 


class Model():
    '''
    class for a NN model 
    '''
    
    def __init__(self, weights, biases, alpha=0.0):
        
        self.weights = weights
        self.biases = biases
        self.num_layers = len(weights)
        self.alpha = alpha 
                          
    def propagate(self, points):
        '''
        propagates a list of points matching the dimension of the model's input layer
        through the model
        
        Parameters:
            points: list of numpy arrays
                    The points to be propagated
        Returns:
            output_points: list of numpy arrays
                    The propagated points 
        '''
        output_points = []
        for point in points:
            for i in range(0, self.num_layers):
                point = np.matmul(self.weights[i], point) + self.biases[i]
               
                if i < self.num_layers - 1:
                    for j in range(0, self.weights[i].shape[0]):
                        if point[j]<0:
                            point[j] = self.alpha * point[j]
        
            output_points.append(point)
        
        return output_points 


class Cube():
    
    '''
    class for n-dimensional cube
    '''
    
    def __init__(self, dimension):
        '''
        creates unit cube of the given dimension

        Parameters
        ----------
        dimension : integer
            dimension 

        Returns
        -------
        None.

        '''
        self.dimension = dimension
        
        points = []
        ps = powerset(range(self.dimension))
        for subset in ps:
            point = np.zeros(self.dimension)
            for l in subset:
                point[l] = 1
            points.append(point)
        
        self.points = points
        
        conns = []
        for k, pt1 in enumerate(self.points):
            for l, pt2 in enumerate(self.points[k+1:]):
                if np.sum(abs(pt1-pt2)) == 1:
                    conns.append((k, k+l+1))
        
        self.conn = set(conns)
    
    def translate(self, b, dim):
        '''
        translates the cube in a given dimension
        
        Parameters:
            dim: integer
                The dimension in which is translated
                enumeration of dim starts at 1
            b: integer
                The scalar by which entry of each vertex in dimension is raised
        '''
        for pt in self.points:
            pt[dim-1] += b
            
        
    def stretch(self, a, dim):
        '''
        stretches the cube in a given dimension by a given factor
        
        Parameters:
            dim: integer
                The dimension in which the cube is stretched
                enumeration of dim starts at 1
            a: integer
                The factor by which the cube is stretched along this dimension
                
        '''
        for pt in self.points:
            pt[dim-1] *= a


def powerset(iterable):
    '''
    determines the power set of an iterable 
    '''
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    
def loadModel(file, alpha=0.00):
    '''
    loads a NN model saved in a .pt-file and creates and instance of class Model
    
    Parameters:
        file: String
            Path of the file containing the saved model
        alpha: float
            Parameter for LeakyReLu activation function of the created model
            Default is 0.00
    
    Returns:
        model: instance of class Model
            The initialized model
    '''
    
    param_dict = torch.load(file) 
    w = []
    b= []
    num_layers = len({int(item.split("_")[1]) for item in param_dict})
    for i in range(num_layers):
            w.append(param_dict["weights_" + str(i)].numpy())
            b.append(param_dict["bias_" + str(i)].numpy())

    model = Model(w, b, alpha=alpha)
    return model 

def loadModel_AS(file, alpha=0):
    '''
    creates an instance of class Model from a file containing a saved model 
    '''
    
    param_dict = file
    w = []
    b= []
    try:
        num_layers = len({int(item.split("_")[1]) for item in param_dict})
    except:
        num_layers = len({int(re.findall(r'\d+', item)[0]) for item in param_dict})

    for i in range(num_layers):
        try:
            w.append(param_dict["weights_" + str(i)].numpy())
            b.append(param_dict["bias_" + str(i)].numpy())
        except:
            w.append(param_dict["fc" + str(i+1) + ".weight"].numpy())
            b.append(param_dict["fc" + str(i+1) + ".bias"].numpy())
    model = Model(w, b, alpha=alpha)
    return model 


def getLinearSegments(model, cpu, alpha = 0.0):  
    '''
    determine the linear segments of a given NN model with LeakyReLu activation
    
    Parameters:
        model: instance of class Model
            the NN model
        cpu: instance of class CPU
            A union of convex polytopes for which the linear segments of model
            shall be determined
        alpha: float
            The parameter of the LeakyReLu activation function of the model
    
    Returns:
        cpu: instance of class CPU
            The convex polytope union obtained containing the linear segments
            of the model with respect to the input CPU
    '''
    
   
    for i in range(0, model.num_layers):
        cpu.linear_transform(model.weights[i], model.biases[i])
        
        dimension = model.weights[i].shape[0]
        if i in range(0, model.num_layers-1):
            for j in range(0, dimension):
                cpu.divide_by_activation(j+1, alpha)
        
    return cpu 


def get_restricted_cpu(model, cpu, cert_vec = None, output_vec = None, alpha = 0.0):
    '''
    determine the linear segments of the restriction of model to a subspace of 
    its input space
    
    Parameters:
        model: instance of class Model
            The NN model
        cpu: instance of class CPU
            A union of convex polytopes for which the linear segments of model
            shall be determined, the dimension of cpu must match the dimension 
            of the subspace to which the model is restricted
        cert_vec: list
            A list containing float values for certain entries and None for
            uncertain entries; the model is restricted to the dimensions given
            by None; number of None entries must match dimension of cpu
            Default is None. 
        output_vec: list
            Default is None 
        alpha: float
            The parameter of the LeakyReLu activation function of the model
    
    Returns:
        cpu: instance of class CPU
            The convex polytope union obtained containing the linear segments
            of the model with respect to the input CPU and the certain values
            given in cert_vec 
        
    '''
    
    adjusted_weights = copy.deepcopy(model.weights)
    adjusted_biases = copy.deepcopy(model.biases)
        
    if cert_vec is not None:
        uncer_idx = [i for i,v in enumerate(cert_vec) if v == None]
        
        if len(uncer_idx) != len(cpu.input_points_union[0]):
            print(uncer_idx,len(cpu.input_points_union[0]))
            raise ValueError("ill-matched certainty dimensions")
        
        w1_shape = model.weights[0].shape
        new_weight = np.zeros((w1_shape[0], len(uncer_idx)))
        
        for i in range(w1_shape[0]):
            none_count = 0
            for j, v in enumerate(cert_vec):
                if v is None:
                    new_weight[i][none_count] = model.weights[0][i][j]
                    none_count +=1
                else:
                    adjusted_biases[0][i] += model.weights[0][i][j]*v
                    
        adjusted_weights[0] = new_weight
    
    if output_vec is not None:
        out_cer_idx = [i for i,v in enumerate(output_vec) if v != None]
        
        new_out_weight = np.zeros((len(out_cer_idx), model.weights[-1].shape[1]))
        new_out_bias = np.zeros(len(out_cer_idx),)
        new_out_weight = model.weights[-1][out_cer_idx]
        new_out_bias = model.biases[-1][out_cer_idx]
        
        adjusted_weights[-1] = new_out_weight
        adjusted_biases[-1] = new_out_bias
        
    for i in range(0, model.num_layers):
        cpu.linear_transform(adjusted_weights[i], adjusted_biases[i])
        
        if i in range(0, model.num_layers-1):
            dimension = adjusted_weights[i].shape[0]
            for j in range(0, dimension):
                cpu.divide_by_activation(j+1, alpha)

    return cpu 



