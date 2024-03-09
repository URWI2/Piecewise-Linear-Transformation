
import copy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import orth
import cdd
import itertools
from scipy.spatial import ConvexHull


def find_int_area(A, index_to_pt_list, lin_map, y):
    '''
    helper function for PLT
    
    Computes the polytope given by the intersection of the preimage of y under 
    the map lin_map and the polytope in the input space A
    
    Paratemeter:
        A: list
            self.points of the polytope in question
        index_to_pt_list: list of numpy arrays
            The list containing all points (and possibly more) the polytope
            references with the indexes given in A.
        lin_map: tuple (Matrix, numpy array)
            The linear map in question
        y: numpy array
            The point in the output we want to find the PPDF for
    
    Returns:
        poly: Instance of class cdd.Polyhedron
            The polytope given by the intersection described above
    '''

    B, bs = lin_map
    polytope_vertices = [np.append(np.ones(1), index_to_pt_list[j]) for j in A]
    mat = cdd.Matrix(polytope_vertices, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    ineqs = poly.get_inequalities()

    mat1 = cdd.Matrix(ineqs, number_type='float')

    try:
        y = y.reshape((np.array(y).shape[0], 1))
    except IndexError:
        y = y.reshape(1, 1)
    ex_rows = np.append(y - np.reshape(bs, y.shape), -B, axis=1)
    mat1.extend(ex_rows, linear=True)

    mat1.rep_type = cdd.RepType.INEQUALITY
    poly_flag = True

    while poly_flag == True:
        try:

            poly = cdd.Polyhedron(mat1)
            poly_flag = False
        except:
            np_mat = np.array(mat1)
            np_mat_1 = np_mat[:-ex_rows.shape[0]]
            np_mat_2 = np_mat[-ex_rows.shape[0]:]

            np_mat_2[-1] *= 1.1
            mat1 = cdd.Matrix(np_mat_1, number_type='float')
            mat1.extend(np_mat_2, linear=True)
            mat1.rep_type = cdd.RepType.INEQUALITY
    gens = poly.get_generators()
    np_gens = np.array(gens)
    if np_gens.shape[0] == 0:
        return None
    else:
        return poly


def get_int_volume(A, index_to_pt_list, lin_map, y):
    '''
    helper function for PLT 
    
    Computes the volume of the polytope given by the intersection of the 
    preimage of y under the map lin_map and the polytope in the input space A
    
    Paratemeter:
        A: list
            self.points of the polytope in question
        index_to_pt_list: list of numpy arrays
            The list containing all points (and possibly more) the polytope
            references with the indexes given in A.
        lin_map: tuple (Matrix, numpy array)
            The linear map in question
        y: numpy array
            The point in the output we want to find the PPDF for
    
    Returns:
        ch.volume: float
            The desired volume
    '''

    if type(y) == np.float64:
        y = np.reshape(y, (1, 1))
    else:
        y = np.reshape(y, (len(y), 1))
    B, bs = lin_map
    B_pi = np.linalg.pinv(B)
    Z_orth = orth(B_pi)

    polytope_vertices = [np.append(np.ones(1), index_to_pt_list[j]) for j in A]
    mat = cdd.Matrix(polytope_vertices, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    ineqs = poly.get_inequalities()

    mat1 = cdd.Matrix(ineqs, number_type='float')
    ex_rows = np.append(y - np.reshape(bs, y.shape), -B, axis=1)
    mat1.extend(ex_rows, linear=True)
    mat1.rep_type = cdd.RepType.INEQUALITY
    poly_flag = True

    while poly_flag == True:
        try:

            poly = cdd.Polyhedron(mat1)
            poly_flag = False
        except:
            np_mat = np.array(mat1)
            np_mat_1 = np_mat[:-ex_rows.shape[0]]
            np_mat_2 = np_mat[-ex_rows.shape[0]:]
            np_mat_2[-1] *= 1.1
            mat1 = cdd.Matrix(np_mat_1, number_type='float')
            mat1.extend(np_mat_2, linear=True)
            mat1.rep_type = cdd.RepType.INEQUALITY

    gens = poly.get_generators()
    np_gens = np.array(gens)

    # calculate volume of polytope
    if np_gens.shape[0] == 0 or np_gens.shape[0] == 1:
        return 0.0
    elif np_gens.shape[0] == 2:
        return np.linalg.norm(np_gens[1] - np_gens[0])
    else:
        np_gens = np.delete(np_gens, 0, axis=1)
        normals = np.transpose(Z_orth)

        if np_gens.shape[0] >= np_gens.shape[1]:
            ch = ConvexHull(np_gens)
            return ch.volume

        else:
            di = len(normals[0])

            for nor in normals:
                np_gens = np.append(np_gens, [np_gens[0]+di*nor], 0)
                di -= 1
            ch = ConvexHull(np_gens)

            return ch.volume


def propagate_adf(adf_method, model):
    '''
    propagation of Gaussian through model using ADF 
    '''
    for i in range(0, model.num_layers):
        adf_method.transform_linear(model.weights[i], model.biases[i])
        if i < model.num_layers - 1:
            adf_method.activation()
    return adf_method.cur_mean, adf_method.cur_var


def comparePDFs(pdfs, our_pdfs, p, width):
    '''
    calculate difference between two pdfs given by histograms on identical bin grids
    in the Lp-norm

    Parameters:
        pdfs: numpy array
            probability masses of first distribution for each bin in a histogram
        our_pdfs: numpy_array
            probability masses of second distribution for each bin in a histogram
        width: float
            volume of a bin in the histogram
        p: float
            Parameter p for Lp-norm to calculate

    Returns:
        temp_norm: float
            The calculated Lp-norm 
    '''

    temp_norm = ((np.sum(abs(pdfs-our_pdfs)**p))*width)**(1/p)

    return temp_norm


def hellinger(pdfs, our_pdfs, width):
    '''
    calculate difference between two pdfs given by histograms on identical bin grids
    in the Hellinger distance

    Parameters:
        pdfs: numpy array
            probability masses of first distribution for each bin in a histogram
        our_pdfs: numpy_array
            probability masses of second distribution for each bin in a histogram
        width: float
            volume of a bin in the histogram

    Returns:
        hell_norm: float
            The calculated Hellinger distance of the two given distributions 
    '''

    hell_norm = (
        (0.5*np.sum((np.sqrt(pdfs)-np.sqrt(our_pdfs))**2))*width)**(1/2)

    return hell_norm


def fit_into_uncert_vec(x, cert_vec):
    '''
    helper function for prepairing vector containing uncertainty information 
    '''
    temp = copy.copy(cert_vec)
    num = 0
    for i, y in enumerate(temp):
        if y == None:
            temp[i] = x[num]
            num += 1
    return temp


def global_method(distr, model, histogram, cert_vec):
    '''
    Calculation of the piecewise constant approximation of the output distribution
    on a given histogram/ bin grid using PLT as described in the paper 

    Parameters:
        distr: instance of class Distribution
            the distribution of the uncertain input attributes
        model: instance of class Model
            the NN model
        histogram: list of numpy arrays
            Histogram of the output space; the bins are saved in histogram[1]
        cert_vec: list
            A vector containing the certain values of the instance and None
            in place of uncertain variables; number of None entries must 
            match the dimension of distr

    Returns:
        our_histo: numpy array
            Array containing the total probability mass of each bin in the 
            histogram 

    '''
    temp = [distr.pdf(x) for x in distr.cpu.subdivision[2]]
    temp_2 = list(distr.cpu.subdivision)
    temp_2.append(temp)
    distr.cpu.subdivision = tuple(temp_2)
    vol_bin = 1
    for j in range(0, len(histogram[1])):
        vol_bin = vol_bin*(histogram[1][j][1]-histogram[1][j][0])

    missed_the_grid_counter = 0
    our_histo = np.zeros(np.shape(histogram[0]), dtype=float)
    points_to_eval = []

    for k, linseg in enumerate(distr.cpu.subdivision[1]):
        for m, delauney in enumerate(linseg):
            full_dim = False
            if len(distr.cpu.subdivision[2][0]) > 1:
                try:
                    hull = ConvexHull([distr.cpu.subdivision[2][x]
                                      for x in distr.cpu.subdivision[0][k][m]])
                    full_dim = True
                except:
                    pass

                if full_dim:
                    vol = hull.volume

                    edge_vol = vol * (1/len(delauney))

                    for h, edgepoly in enumerate(delauney):
                        center = np.zeros(
                            np.shape(distr.cpu.subdivision[2][edgepoly[0]]))
                        av_pdf = 0
                        for x in edgepoly:
                            center += distr.cpu.subdivision[2][x]
                            av_pdf += distr.cpu.subdivision[3][x]
                        center = center*1/len(edgepoly)
                        av_pdf = av_pdf*1/len(edgepoly)
                        center = fit_into_uncert_vec(center, cert_vec)
                        l = model.propagate([center])

                        points_to_eval.append(
                            (l[0], edge_vol, (k, m, h), av_pdf))

            else:
                hull_gens = [distr.cpu.subdivision[2][x]
                             for x in distr.cpu.subdivision[0][k][m]]
                if len(hull_gens) != 2:
                    print("Not 2 generators")
                else:
                    vol = float(np.abs(hull_gens[1] - hull_gens[0]))
                    edge_vol = vol/len(delauney)
                    for h, edgepoly in enumerate(delauney):
                        center = np.zeros(
                            np.shape(distr.cpu.subdivision[2][edgepoly[0]]))
                        av_pdf = 0
                        for x in edgepoly:
                            center += distr.cpu.subdivision[2][x]
                            av_pdf += distr.cpu.subdivision[3][x]
                        center = center/len(edgepoly)
                        av_pdf = av_pdf*1/len(edgepoly)
                        center = fit_into_uncert_vec(center, cert_vec)
                        l = model.propagate([center])

                        points_to_eval.append(
                            (l[0], edge_vol, (k, m, h), av_pdf))

    for point, e_vol, num, value in points_to_eval:

        # find the bin of the point
        correct_bin = np.zeros(np.shape(point), dtype=int)
        for i in range(0, np.shape(point)[0]):

            temp = point[i]
            temp = temp-histogram[1][i][0]
            temp = temp*(1/(histogram[1][i][1]-histogram[1][i][0]))

            temp = int(temp)
            correct_bin[i] += temp
        correct_bin = tuple(correct_bin)

        try:
            our_histo[correct_bin] += 0
            method = value
            pdf_val = method*(e_vol/vol_bin)
            our_histo[correct_bin] += pdf_val
        except IndexError:
            missed_the_grid_counter += 1


    return our_histo


def make_histo_grid(histo):
    '''
    helper function for grid creation 
    '''
    histo_grid = iterative_histo_grid([[]], copy.copy(histo))
    histo_grid = [np.array(x) for x in histo_grid]
    histo_grid = iterative_grid_from_list(
        histo_grid, bin_num=len(histo[0]), dim=len(histo))
    return histo_grid


def iterative_grid_from_list(l, bin_num, dim):
    '''
    helper function for grid creation 
    '''
    if dim == 1:
        return l
    else:
        temp = []
        for i in range(bin_num):
            temp_2 = []
            for k in range(int(i*(len(l)/bin_num)), int((i+1)*(len(l)/bin_num))):
                temp_2.append(l[k])
            temp.append(iterative_grid_from_list(temp_2, bin_num, dim-1))
        return temp


def iterative_histo_grid(histo_grid, gridpoints):
    '''
    helper function for grid creation 
    '''
    if gridpoints == []:
        return histo_grid
    else:
        temp = []
        for x in histo_grid:
            for i in range(len(gridpoints[0])):
                y = copy.copy(x)
                y.append(gridpoints[0][i])
                temp.append(y)
        return iterative_histo_grid(temp, gridpoints[1:])


class Distribution():
    '''
    class for a distribution on a convex polytope union 
    '''

    def __init__(self, cpu, model=None, pdf=None, distr_type="custom", *args, **kwargs):
        
        '''
        initializes a distribution on a convex polygon union given by a normal
        distribution, GMM or Scipy KDE
        A model is saved along with the distribution for later propagation
        of distribution through this model using PLT 
        '''

        self.cpu = cpu
        self.dim = len(cpu.input_points_union[0])
        self.output_dim = len(cpu.current_points_union[0])
        self.distr_type = distr_type
        self.model = model

        # self.min_max contains minimum and maximum values of the transformed (by the piecewise linear function)
        # input space in each dimension
        pdf_max_min = []
        for i in range(self.output_dim):
            dim_min = min([entry[i]
                          for entry in self.cpu.current_points_union])
            dim_max = max([entry[i]
                          for entry in self.cpu.current_points_union])
            pdf_max_min.append([dim_min, dim_max])
        self.min_max = pdf_max_min

        if pdf is not None:
            self.pdf = pdf

        # normal distribution as input
        elif self.distr_type == "normal":

            if 'mean' in kwargs:
                mean = kwargs['mean']
            else:
                mean = np.zeros(self.dim)

            if 'cov' in kwargs:
                cov = kwargs['cov']
            else:
                cov = np.identity(self.dim)

            self.dis = multivariate_normal(mean=mean, cov=cov)
            self.pdf = self.dis.pdf

        # Gaussian Mixture Model (GMM) as input
        elif self.distr_type == "GMM":
            if 'GMM' in kwargs:
                self.dis = kwargs['GMM']
                self.pdf = self.GMM_pdf

        # initialize values on subdivision, if one is given
        if self.cpu.subdivision is not None:
            pass

    def GMM_pdf(self, x):
        '''
        calculate PDF values for a GMM distribution 
        '''
        pdf_val = np.zeros((len(x), self.dis.n_components))
        for i in range(self.dis.n_components):
            pdf_val[:, i] = multivariate_normal.pdf(
                x, mean=self.dis.means_[i], cov=np.diagflat(self.dis.covariances_[i]))
        print(pdf_val.shape)
        print(self.dis.weights_)
        pdf_val = pdf_val @ self.dis.weights_
        return pdf_val

    # HAVE TURNED RESTRICTED NORMAL DISTRIBUTION SAMPLES INTO ARRAY NOW; HAS NOT BEEN TESTED
    def sample(self, nr_samples, input_cube=None):
        '''
        sample from the distribution on a given 
        Normal distributions, GMM distribution and Scipy KDE kernels are supported 

        Parameters:
            nr_samples: integer
                The number of samples to draw from the distribution
            input_cube: instance of class CPU
                a cube to which the sampling from a normal distribution can
                be restricted
                Restricting sampling to an input_cube is only implemented if
                the distribution is given by a normal distribution; GMM 
                sampling is only implemented if input_cube is None, for 
                sampling from KDE the parameter input_cube does not influence
                the sampling result

        Returns:
            sample_array: numpy array
                The drawn samples 

        '''
        if self.distr_type == "GMM":
            if input_cube is None:
                sample_array = self.dis.sample(nr_samples)[0]

            return sample_array
        if self.distr_type == "normal":
            if input_cube is None:
                sample_array = self.dis.rvs(size=nr_samples)
            else:
                cube_min_max = []
                for i in range(input_cube.dimension):
                    dim_min = min([entry[i] for entry in input_cube.points])
                    dim_max = max([entry[i] for entry in input_cube.points])
                    cube_min_max.append([dim_min, dim_max])
                sample_array = []
                i = 0
                while i in range(nr_samples):
                    samp = self.dis.rvs(size=1)
                    if samp.size == 1:
                        samp = [samp]
                    samp_flag = True
                    for j in range(input_cube.dimension):
                        if samp[j] < cube_min_max[j][0] or samp[j] > cube_min_max[j][1]:
                            samp_flag = False
                    if samp_flag == True:
                        sample_array.append(samp)
                        i += 1
                sample_array = np.array(sample_array)
        if self.distr_type == "custom":
            sample_array = self.pdf.resample(nr_samples)
            sample_array = sample_array.transpose()

            return sample_array


    def local_method(self, x, k=None, e=None):
        '''
        calculate the PDF values for a list of points in the output space
        using PLT (following formula (5) from the paper)

        Parameters:
            x: list of numpy arrays
                list of points in the output space
                If x is given by a 1D-array representing one output point 
                instead of a list, this is equivalent to setting x to a list
                containing only this point
            k: integer
                parameter k for the edgewise subdivision
            e: integer
                threshold for edgewise subdivision 

        Returns:
            prop_pdf_vals: list of floats
                the PDF values for the points in x 
        '''
        if self.cpu.subdivision is None:
            self.cpu.subdivide(k=k, epsilon=e)
        if type(x) is not list:
            x = list(x)

        prop_pdf_vals = []
        poly_list = self.cpu.checkPolytopes(x, eps=10**(-9))

        if self.cpu.lin_seg_preps is None:
            self.cpu.prep_lin_seg_for_eval()

        for i, polys_for_val in enumerate(poly_list):
            pdf_val = 0
            for j in polys_for_val:
                j_prep = self.cpu.lin_seg_preps[j]
                W, b = self.cpu.polytopes[j].aff_lin

                if j_prep[0] == 0:
                    W_inv = j_prep[1]
                    det_W_inv = j_prep[2]
                    x_preim = W_inv@(x[i] - b)
                    x_pdf = self.pdf(x_preim)*det_W_inv
                    pdf_val += x_pdf

                if j_prep[0] == 1:
                    RT = j_prep[3]
                    b_tilde = j_prep[4]
                    W_tilde_inv = j_prep[1]
                    det_W_tilde_inv = j_prep[2]

                    x_tilde = RT@x[i]
                    x_preim = W_tilde_inv@(x_tilde - b_tilde)
                    x_pdf = self.pdf(x_preim)*det_W_tilde_inv
                    pdf_val += x_pdf

                if j_prep[0] == 2 or j_prep[0] == 3:

                    det_W_tilde_inv = j_prep[1]

                    for n, tri in enumerate(self.cpu.subdivision[0][j]):
                        if find_int_area(tri, self.cpu.subdivision[2], (W, b), x[i]) is not None:
                            for m, edge_sim in enumerate(self.cpu.subdivision[1][j][n]):
                                # find intersection volume
                                area = get_int_volume(
                                    edge_sim, self.cpu.subdivision[2], (W, b), x[i])

                                if area != 0.0:
                                    pdf_value = np.mean(
                                        self.pdf([self.cpu.subdivision[2][u] for u in edge_sim]))
                                    x_pdf = area*pdf_value*det_W_tilde_inv
                                    pdf_val += x_pdf

            prop_pdf_vals.append(pdf_val)

        return prop_pdf_vals

    def integrateOverCube(self):
        '''
        calculate integral of a normal distribution over a rectangle in the input
        space using cumulative density functions 

        Returns:
            integral: float
                The value of the integral of the distribution over its cpu
                This only works if the distribution is given by a normal 
                distribution and the cpu is a rectangle 
        '''

        if self.distr_type != "normal":
            print("Function only works for normal distribution")
            return

        # get boundaries of the input rectangle in every dimension
        lowers = []
        uppers = []
        for i in range(self.dim):
            dim_entries = [point[i] for point in self.cpu.input_points_union]
            lowers.append(min(dim_entries))
            uppers.append(max(dim_entries))

        integral = self.dis.cdf(np.array(uppers))
        print("Integral", integral)
        for i in range(1, self.dim + 1):
            for comb in itertools.combinations(range(self.dim), i):
                print(comb)
                point = np.array(uppers)

                for j in comb:
                    point[j] = lowers[j]
                print(point, (-1)**i)
                integral += ((-1)**i)*self.dis.cdf(point)
                print(integral)

        return integral

    def normalizePDF(self):
        '''
        calculates the integral of the distribution on its cpu if the 
        distribution is normal and the cpu is a rectangle and normalizes
        the PDF without changing self.dis

        Returns:
            new_PDF: function
                The new value of self.pdf after normalizing the PDF function
                This only works if self is a normal distribution and self.cpu
                a rectangle 
        '''

        volume = self.integrateOverCube()

        def newPDF(x):
            old = self.dis.pdf(x)
            return (1/volume)*old

        self.pdf = newPDF

        return newPDF


    def find_baseline_histo(self, cert_vec=None, out_dim=None, start_params=(1000000, 100),
                            samp_nr_mult=5, bin_nr_mult=2, bin_it=10, threshold=0.01):
        '''
        Find baseline histogram in the output space 
        using sampling (for the propagation of 1 instance) iteratively
        A propagated PD is obtained by drawing a given number of samples and
        using a histogram with a given number of bins
        These numbers are iteratively increased until the L1-difference between
        the last two histograms is lower than a given threshold 

        Parameters:
            cert_vec: list
                List containing the certain attribute values of the instance
                and None in place of uncertain attributes
            out_dim: integer or None
                the output dimension
            start_params: tuple of 2 integers
                start_params[0]: the number of samples drawn in the first iteration
                start_params[1]: number of bins per dimension in the first iteration
            samp_nr_mult: int
                The multiplier by which the number of samples drawn is increased
            bin_nr_mult: int
                The multiplier by which the number of bins per dimension is increased
            bin_it: int
                Number of iterations after which bin number is increased
            threshold: float
                The threshold used as stopping criterion
                Default is 0.01
            
            
        Returns:
            out_samps_dim_1, histo_1, histo_diff, current_params 
                
        '''
        samp_nr_1 = start_params[0]
        bins_nr_1 = start_params[1]
        histo_diff = threshold + 1

        samps_1 = np.array(self.sample(samp_nr_1))

        if len(samps_1.shape) == 1:
            samps_1 = np.reshape(samps_1, (len(samps_1), 1))

        if cert_vec is not None:
            uncer_idx = [i for i, v in enumerate(cert_vec) if v == None]
            cer_idx = [i for i, v in enumerate(cert_vec) if v != None]
            sample_vec_1 = np.zeros((samp_nr_1, len(cert_vec)))
            for idx, val in enumerate(cert_vec):
                if idx in cer_idx:
                    sample_vec_1[:, idx] = val
                else:
                    sample_vec_1[:, idx] = samps_1[:, uncer_idx.index(idx)]
        else:
            sample_vec_1 = samps_1

        out_samps_1 = self.model.propagate(sample_vec_1)

        if out_dim is not None:
            out_samps_dim_1 = np.array(out_samps_1)[:, out_dim]
        else:
            out_samps_dim_1 = np.array(out_samps_1)

        while_count = 1
        while histo_diff >= threshold:
            histo_diff = 0
            samp_nr_2 = samp_nr_mult * samp_nr_1
            if while_count % bin_it == 0:
                bins_nr_2 = bin_nr_mult * bins_nr_1
            else:
                bins_nr_2 = bins_nr_1
            print("new histo params: histo_1", (samp_nr_1, bins_nr_1),
                  " histo_2 ", ((samp_nr_2, bins_nr_2)))

            samps_2 = np.array(self.sample(samp_nr_2))
            if len(samps_2.shape) == 1:
                samps_2 = np.reshape(samps_2, (len(samps_2), 1))

            if cert_vec is not None:
                uncer_idx = [i for i, v in enumerate(cert_vec) if v == None]
                cer_idx = [i for i, v in enumerate(cert_vec) if v != None]

                sample_vec_2 = np.zeros((samp_nr_2, len(cert_vec)))

                for idx, val in enumerate(cert_vec):
                    if idx in cer_idx:
                        sample_vec_2[:, idx] = val
                    else:
                        sample_vec_2[:, idx] = samps_2[:, uncer_idx.index(idx)]
            else:
                sample_vec_2 = samps_2

            out_samps_2 = self.model.propagate(sample_vec_2)

            if out_dim is not None:
                out_samps_dim_2 = np.array(out_samps_2)[:, out_dim]
            else:
                out_samps_dim_2 = np.array(out_samps_2)

            histo_2 = np.histogramdd(
                out_samps_dim_2, bins=bins_nr_2, density=True)

            histo_range_list = []
            for h_dim in histo_2[1]:
                histo_range_list.append((h_dim[0], h_dim[-1]))

            histo_1_comp = np.histogramdd(
                out_samps_dim_1, bins=bins_nr_1, range=histo_range_list, density=True)
            histo_width_list = []
            for dim_width in histo_2[1]:
                histo_width_list.append(dim_width[1] - dim_width[0])
            np_width_arr = np.array(histo_width_list)
            bin_vol = np.prod(np_width_arr)

            histo_1_comp_data = histo_1_comp[0]
            if while_count % bin_it == 0:
                for dim_i in range(len(histo_1_comp_data.shape)):
                    histo_1_comp_data = np.repeat(
                        histo_1_comp_data, bin_nr_mult, axis=dim_i)

            diff = np.sum(np.abs(histo_1_comp_data - histo_2[0]))

            histo_diff = diff * bin_vol
            current_params = (samp_nr_1, bins_nr_1)
            print(histo_diff, current_params, "<->", (samp_nr_2, bins_nr_2))
            histo_1 = histo_2
            out_samps_dim_1 = out_samps_dim_2
            samp_nr_1 = samp_nr_2
            bins_nr_1 = bins_nr_2
            while_count += 1
        return out_samps_dim_1, histo_1, histo_diff, current_params
