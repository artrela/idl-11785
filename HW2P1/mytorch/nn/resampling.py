import numpy as np
import pdb


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        
        # make a matrix of 0's
        Win = A.shape[2]
        Wout = self.upsampling_factor * (Win-1) + 1
        Z = np.zeros((A.shape[0], A.shape[1], Wout))  
        
        # fill every K-1 idx of Z with A
        self.fill_idx = np.array([i + i*(self.upsampling_factor-1) for i in range(Win)])
        Z[:, :, self.fill_idx] = A
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # Take out the K-1th element for dLdA
        dLdA = dLdZ[:, :, self.fill_idx]  # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # we want every kth element of A
        self.Win = A.shape[-1]
        self.fill_idx = np.array([i for i in range(0, self.Win, self.downsampling_factor)])
        Z =  A[:, :, self.fill_idx]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # now we need to fill every kth element of dLdA
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.Win))
        dLdA[:, :, self.fill_idx] = dLdZ
        
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = None  # TODO

        return NotImplemented


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = None  # TODO

        return NotImplemented
