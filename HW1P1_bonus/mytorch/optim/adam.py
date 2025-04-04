# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import pdb
import numpy as np

class Adam():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.l = model.layers[::2] # every second layer is activation function
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]

    def step(self):
        
        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # TODO: Calculate updates for weight
            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * layer.dLdW 
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * layer.dLdW**2
            
            # increases this affect early on when mean is zero or small (dont save in momentum)
            m_W = self.m_W[layer_id] / ( 1 - self.beta1**self.t )
            v_W = self.v_W[layer_id] / ( 1 - self.beta2**self.t )
            
            # TODO: calculate updates for bias
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.dLdb 
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * layer.dLdb**2
            
            m_b = self.m_b[layer_id] / ( 1 - self.beta1**self.t)
            v_b = self.v_b[layer_id] / ( 1 - self.beta2**self.t)

            # TODO: Perform weight and bias updates
            layer.W -= self.lr * m_W / np.sqrt(v_W + self.eps)
            layer.b -= self.lr * m_b / np.sqrt(v_b + self.eps)

