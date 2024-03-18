import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        self.r = self.r_act.forward( self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh )
        self.z = self.z_act.forward( self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh )
        self.n = self.h_act.forward( self.Wnx @ x + self.bnx + self.r*(self.Wnh @ h_prev_t + self.bnh) )
        h_t = (1 - self.z)*self.n + self.z*h_prev_t
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        # raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.
        
        # hidden = h_t-1
        
        # Forward Eq.1.
        dz_t = delta * ( -self.n + self.hidden )
        assert dz_t.shape == self.z.shape
        
        dn_t = delta * (1 - self.z)
        assert dn_t.shape == self.n.shape
        
        # Forward Eq.2.
        self.dWnx = self.h_act.backward(dn_t)[:, np.newaxis] @ self.x[np.newaxis, :]
        self.dbnx = self.h_act.backward(dn_t) 
        
        dr_t = self.h_act.backward(dn_t) * (self.Wnh @ self.hidden + self.bnh)
        assert dr_t.shape == self.r.shape
        
        self.dWnh = self.h_act.backward(dn_t)[:, np.newaxis] * self.r[:, np.newaxis] @ self.hidden[np.newaxis, :]
        self.dbnh = self.h_act.backward(dn_t) * self.r
        
        # Forward Eq.3.
        self.dWzx = self.z_act.backward(dz_t)[:, np.newaxis] @ self.x[np.newaxis, :]
        self.dbzx = self.z_act.backward(dz_t)
        
        self.dWzh = self.z_act.backward(dz_t)[:, np.newaxis] @ self.hidden[np.newaxis, :]
        self.dbzh = self.z_act.backward(dz_t)
        
        # Forward Eq.4.
        self.dWrx = self.r_act.backward(dr_t)[:, np.newaxis] @ self.x[np.newaxis, :]
        self.dbrx = self.r_act.backward(dr_t)
        
        self.dWrh = self.r_act.backward(dr_t)[:, np.newaxis] @ self.hidden[np.newaxis, :]
        self.dbrh = self.r_act.backward(dr_t)
        
        # multiple terms
        dnt_dx = self.Wnx.T @ self.h_act.backward(dn_t)
        dzt_dx = self.Wzx.T @ self.z_act.backward(dz_t)
        drt_dx = self.Wrx.T @ self.r_act.backward(dr_t)
        
        dx = dnt_dx + dzt_dx + drt_dx
        
        dht_dhprev = delta*self.z
        dnt_dhprev = self.h_act.backward(dn_t) * self.r @ self.Wnh
        dzt_dhprev = self.z_act.backward(dz_t) @ self.Wzh
        drt_dhprev = self.r_act.backward(dr_t) @ self.Wrh 
        
        dh_prev_t = dht_dhprev + dnt_dhprev + dzt_dhprev + drt_dhprev
        
        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
