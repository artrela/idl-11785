import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        
        N, Cin, self.Hin, self.Win = A.shape
        self.Ashape = A.shape
        
        outputH = self.Hin - self.kernel_size + 1
        outputW = self.Win - self.kernel_size + 1
        
        Z = np.zeros((N, self.out_channels, outputH, outputW))
            
        # slice of the batch size
        for h in range(outputH):
            for w in range(outputW):
                A_slice = A[:, :, h:h+self.kernel_size, w:w+self.kernel_size]
                Z[:, :, h, w] = np.tensordot(A_slice, self.W, ([1, 2, 3], [1, 2, 3]))
                    
        return Z + np.expand_dims(self.b, (0, 2, 3))

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        self.dLdW = np.zeros(self.W.shape)
        for h in range(self.W.shape[2]):
            for w in range(self.W.shape[3]):
                
                A_slice = self.A[:, :, h:h+dLdZ.shape[2], w:w+dLdZ.shape[3]]                
                self.dLdW[:, :, h, w] = np.tensordot(dLdZ, A_slice, ([0, 2, 3], [0, 2, 3]))
         
        
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # TODO
        
        padded_dLdZ = np.pad(dLdZ, ((0,), (0,), (self.kernel_size-1,), (self.kernel_size-1,)))
        dLdA = np.zeros(self.Ashape)  # TODO

        for h in range(self.Hin):
            for w in range(self.Win):
                
                slice_dLdZ = padded_dLdZ[:, :, h:h+self.kernel_size, w:w+self.kernel_size]
                W_flip = np.flip(self.W, axis=(2, 3))

                dLdA[:, :, h, w] = np.tensordot(slice_dLdZ, W_flip, ([1, 2, 3], [0, 2, 3]))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        padded_input = np.pad(A, ((0,), (0,), (self.pad,), (self.pad,)))

         # Call Conv2d_stride1
        conv_input = self.conv2d_stride1.forward(padded_input)

        # downsample
        Z = self.downsample2d.forward(conv_input)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        ds_back = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(ds_back)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:dLdA.shape[-2]-self.pad, self.pad:dLdA.shape[-1]-self.pad]
        
        return dLdA
