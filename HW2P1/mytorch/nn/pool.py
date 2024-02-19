import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = np.zeros((A.shape[0], A.shape[1], A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1))
        self.cache = np.empty(Z.shape, dtype=list)
        self.A_shape = A.shape
                
        for z in range(Z.size):
            
            b, c, i, j = np.unravel_index(z, Z.shape)
                
            A_slice = A[b, c, i:i+self.kernel, j:j+self.kernel]
            
            max_i, max_j = np.unravel_index(np.argmax(A_slice), A_slice.shape)

            Z[b, c, i, j] = A_slice[max_i, max_j]
            self.cache[b, c, i, j] = [max_i + i, max_j + j]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = np.zeros(self.A_shape)
        
        for idx in range(self.cache.size):
            
            b, c, i, j = np.unravel_index(idx, self.cache.shape)
            max_idxs = self.cache[b, c, i, j]
            
            dLdA[b, c, max_idxs[0], max_idxs[1]] += dLdZ[b, c, i, j]
            
        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = np.zeros((A.shape[0], A.shape[1], A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1))
        self.A_shape = A.shape
                
        for z in range(Z.size):
            
            b, c, i, j = np.unravel_index(z, Z.shape)
                
            A_slice = A[b, c, i:i+self.kernel, j:j+self.kernel]
            
            Z[b, c, i, j] = np.mean(A_slice)
            
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.A_shape)
        
        for idx in range(dLdZ.size):
            
            b, c, i, j = np.unravel_index(idx, dLdZ.shape)
            
            dLdA[b, c, i:i+self.kernel, j:j+self.kernel] += ( self.kernel ** -2) * dLdZ[b, c, i, j]
            
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = self.maxpool2d_stride1.forward(A)
        
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = self.downsample2d.backward(dLdZ)
        
        dLdA = self.maxpool2d_stride1.backward(dLdA)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = self.meanpool2d_stride1.forward(A)
        
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = self.downsample2d.backward(dLdZ)
        
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        
        return dLdA
