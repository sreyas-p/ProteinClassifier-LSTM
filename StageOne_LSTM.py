import random
import numpy as np
import math

#loss function
def sigmoid(x): 
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values): 
    return values*(1-values)

#working memory loss function
def tanh_derivative(values): 
    return 1. - values ** 2

#generating weights
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

#class function --> training weights 
class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct)
        
        #stacking x(present input xt) and h(t-1)
        xc = np.hstack((x,  h_prev))
        #dot product of Wf(forget weight matrix and xc +bias)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        #finally multiplying forget_gate(self.state.f) with previous cell state(s_prev) 
        #to get present state.
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        
        #to calculate the output state --> compute the relevancy of the previous output (i.e working memory) Dot Product
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        #output state h --> long term memroy (update values to memory vector array)
        self.state.h = self.state.s * self.state.o
                
