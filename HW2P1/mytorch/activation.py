import numpy as np
from scipy.special import erf # type: ignore

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):

        self.A = 1 / ( 1 + np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (self.A * ( 1 - self.A))

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    
    def forward(self, Z):

        self.A = np.tanh(Z) #(np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (1 - self.A*self.A)

        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    
    def forward(self, Z):

        self.A = np.maximum(Z, np.zeros_like(Z))

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * np.where(self.A > 0, 1, 0)

        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    
    def forward(self, Z):

        self.A = 0.5*Z * ( 1 + erf(Z / np.sqrt(2)))
        self.Z = Z

        return self.A

    def backward(self, dLdA):
        
        dAdZ = 0.5*(1 + erf(self.Z / np.sqrt(2))) + \
                (self.Z / np.sqrt(2*np.pi))*np.exp((-self.Z**2)/2)

        dLdZ = dLdA * dAdZ

        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        
        self.A = np.divide(np.exp(Z), 
                           np.expand_dims(np.sum( np.exp(Z), axis=1), 1))

        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N, C = dLdA.shape

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C)) 

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C)) 

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    
                    if m == n:
                        J[m,n] = self.A[i, m] * (1 - self.A[i, m]) 
                    else:
                        J[m,n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i, :] @ J

        return dLdZ