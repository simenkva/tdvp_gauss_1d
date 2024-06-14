from sympy import *
import numpy as np
from icecream import ic
from fft_tdse.fouriergrid import fftgrid
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import expm
import matplotlib.pyplot as plt

# set display to pretty printing if not in a notebook
try:
    get_ipython()
        
except:
    print('Not in a notebook')
    from sympy.interactive import printing
    printing.init_printing(use_latex=True)
    # define display to pretty print expression
    def display(expr):
        return pprint(expr)


def plot_basis(z, x, gaussians):
    """ Simple utility to plot gaussian basis sets"""
    plt.figure()
    for k in range(z.shape[1]):
        phi = gaussians.gaussian_lamb(x, *z[:, k])
        color = f'C{k}' 
        plt.plot(x, phi.real, label=f'$\Re\phi_{{{k}}}$', color=color)
        plt.plot(x, phi.imag, label=f'$\Im\phi_{{{k}}}$', color=color, linestyle='--')
    plt.legend()
    plt.show()
    
def plot_approx(z, target_fun, x, gaussians, target_label='target', approx_label='approximation'):
    df = LCGauss1D(gaussians)
    df.set_z(z)
    c = df.projection(x, target_fun)
    
    #c_block = (c[:n] + c[n:].conj())/2 # purify c
    #c = np.concatenate([c_block, c_block.conj()])
    approx = df.eval_grid(x, c)
    plt.figure()
    plt.plot(x, target_fun, label=target_label)   
    plt.plot(x, approx, label=approx_label)
    plt.legend()
    plt.figure()
    plt.semilogy(x, np.abs(target_fun - approx), label='abs error')
    plt.semilogy(x, np.abs(approx.imag), label='abs imag')
    plt.legend()
    plt.show()




class Gauss1D:
    def mylambdify(self, f):
        X0 = self.X0
        X1 = self.X1
        X2 = self.X2
        lamb0 = lambdify((X0,X1,X2), f)
        lamb = lambda X: lamb0(*X)
        return lamb
        
    def __init__(self, Q = None, varlist = None, verbose = False):
        # Define symbols
        X0 = Symbol('X_0')
        X1 = Symbol('X_1')
        X2 = Symbol('X_2')
        x = Symbol('x',real=True)
        self.x = x
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2
        

        # The primitive gaussian exponent
        self.Q0 = -X2*x**2 - X1*x - X0

        # The primitive gaussian integral
        self.J_sym = sqrt(pi/X2)*exp(X1**2/(4*X2) - X0)
        self.J = self.mylambdify(self.J_sym)

        # self.J_lamb0 = lambdify((X0,X1,X2), self.J)
        # self.J_lamb = lambda X: self.J_lamb0(*X)
        

        # Get the primitive gaussian normalization constant
        u2,u3 = symbols('u2,u3',real=True)
        u1 = Symbol('u1',positive=True)
        N = 1/sqrt(self.J_sym.subs({X0: 2*u1, X1: 2*u2, X2: 2*u3})) # sympy not so good at re and im
        self.N = N.subs({u1:re(X0),u2:re(X1),u3:re(X2)})

        # Strategy to simplify expression:
        # differentiate, isolate prefactor, expand, and insert prefactor again
        self.mydiff = lambda f,x: expand(diff(f,x)/self.J_sym)*self.J_sym
        self.mydiff2 = lambda f,x: simplify(expand(diff(f*self.J_sym,x))/self.J_sym)

        # Print some info if verbose == True
        self.verbose = verbose

        self.compute_derivatives()

#        self.compute_G_derivs()
        self.compute_Phi_maps(Q, varlist)

    def compute_derivatives(self):
        X0 = self.X0
        X1 = self.X1
        X2 = self.X2
        x = self.x
      
        
        self.J001_sym = self.mydiff(self.J_sym, X2)
        self.J010_sym = self.mydiff(self.J_sym, X1)
        self.J100_sym = self.mydiff(self.J_sym, X0)
        self.J011_sym = self.mydiff(self.J010_sym, X2)
        self.J101_sym = self.mydiff(self.J001_sym, X0)
        self.J110_sym = self.mydiff(self.J100_sym, X1)
        self.J200_sym = self.mydiff(self.J100_sym, X0)
        self.J020_sym = self.mydiff(self.J010_sym, X1)
        self.J002_sym = self.mydiff(self.J001_sym, X2)
        self.J030_sym = self.mydiff(self.J020_sym, X1)
        self.J120_sym = self.mydiff(self.J110_sym, X1)
        self.J021_sym = self.mydiff(self.J020_sym, X2)
        self.J201_sym = self.mydiff(self.J200_sym, X2)
        self.J102_sym = self.mydiff(self.J101_sym, X2)

        # expoential function is common for all integrals -
        # factor it.
        self.fJ001_sym = simplify(self.J001_sym/self.J_sym)
        self.fJ010_sym = simplify(self.J010_sym/self.J_sym)
        self.fJ100_sym = simplify(self.J100_sym/self.J_sym)
        self.fJ011_sym = simplify(self.J011_sym/self.J_sym)
        self.fJ101_sym = simplify(self.J101_sym/self.J_sym)
        self.fJ110_sym = simplify(self.J110_sym/self.J_sym)
        self.fJ200_sym = simplify(self.J200_sym/self.J_sym)
        self.fJ020_sym = simplify(self.J020_sym/self.J_sym)
        self.fJ002_sym = simplify(self.J002_sym/self.J_sym)
        self.fJ030_sym = simplify(self.J030_sym/self.J_sym)
        self.fJ120_sym = simplify(self.J120_sym/self.J_sym)
        self.fJ021_sym = simplify(self.J021_sym/self.J_sym)
        
        


        
        self.J001 = self.mylambdify(self.J001_sym)
        self.J010 = self.mylambdify(self.J010_sym)
        self.J100 = self.mylambdify(self.J100_sym)
        self.J011 = self.mylambdify(self.J011_sym)
        self.J101 = self.mylambdify(self.J101_sym)
        self.J110 = self.mylambdify(self.J110_sym)
        self.J200 = self.mylambdify(self.J200_sym)
        self.J020 = self.mylambdify(self.J020_sym)
        self.J002 = self.mylambdify(self.J002_sym)
        self.J030 = self.mylambdify(self.J030_sym)
        self.J120 = self.mylambdify(self.J120_sym)
        self.J021 = self.mylambdify(self.J021_sym)
        

        self.fJ001 = self.mylambdify(self.fJ001_sym)
        self.fJ010 = self.mylambdify(self.fJ010_sym)
        self.fJ100 = self.mylambdify(self.fJ100_sym)
        self.fJ011 = self.mylambdify(self.fJ011_sym)
        self.fJ101 = self.mylambdify(self.fJ101_sym)
        self.fJ110 = self.mylambdify(self.fJ110_sym)
        self.fJ200 = self.mylambdify(self.fJ200_sym)
        self.fJ020 = self.mylambdify(self.fJ020_sym)
        self.fJ002 = self.mylambdify(self.fJ002_sym)
        self.fJ030 = self.mylambdify(self.fJ030_sym)
        self.fJ120 = self.mylambdify(self.fJ120_sym)
        self.fJ021 = self.mylambdify(self.fJ021_sym)
        
        
        
        if self.verbose:
            display(Eq(Symbol('J'), self.J_sym))
            display(Eq(Symbol('J001'), self.J001_sym))
            display(Eq(Symbol('J010'), self.J010_sym))
            display(Eq(Symbol('J100'), self.J100_sym))
            display(Eq(Symbol('J011'), self.J011_sym))
            display(Eq(Symbol('J101'), self.J101_sym))
            display(Eq(Symbol('J110'), self.J110_sym))
            display(Eq(Symbol('J200'), self.J200_sym))
            display(Eq(Symbol('J020'), self.J020_sym))
            display(Eq(Symbol('J002'), self.J002_sym))

        
        # # Compute the moments of the gaussian up to fourth order
        # self.G1 = -self.mydiff(self.J, X1)
        # self.G2 = -self.mydiff(self.J, X2)
        # self.G3 = -self.mydiff(self.G1, X2)
        # self.G4 = -self.mydiff(self.G2, X2)

        # if self.verbose:
        #     display(Eq(Symbol('J'), self.J))
        #     display(Eq(Symbol('G1'), self.G1))
        #     display(Eq(Symbol('G2'), self.G2))
        #     display(Eq(Symbol('G3'), self.G3))
        #     display(Eq(Symbol('G4'), self.G4))

    # def compute_moments(self):
    #     X0 = self.X0
    #     X1 = self.X1
    #     X2 = self.X2
    #     x = self.x
        
    #     # Compute the moments of the gaussian
    #     self.G1 = -self.mydiff(self.G, X1)
    #     self.G2 = -self.mydiff(self.G, X2)
    #     self.G3 = -self.mydiff(self.G1, X2)
    #     self.G4 = -self.mydiff(self.G2, X2)

    #     if self.verbose:
    #         display(Eq(Symbol('G'), self.G))
    #         display(Eq(Symbol('G1'), self.G1))
    #         display(Eq(Symbol('G2'), self.G2))
    #         display(Eq(Symbol('G3'), self.G3))
    #         display(Eq(Symbol('G4'), self.G4))
                
        
    def compute_Phi_maps(self, Q = None, varlist = None):

        x = self.x
        if Q is None or varlist is None:
            varlist = symbols('a,b,q,p',real=True)

            a,b,q,p = varlist

#            self.Q = -(a + I*b)*(x - q)**2/2 + I*p*(x-q)
            self.Q = -(exp(a) + I*b)*(x - q)**2/2 + I*p*(x-q)
            self.varlist = varlist
            del a,b,q,p
        else:
            self.Q = Q
            self.varlist = varlist
        self.n_par = len(self.varlist)


        if self.verbose:
            print('Exponent:')
            display(Eq(Symbol('Q'),self.Q))
            print('Variable list:')
            display(self.varlist)

        # Save gaussian evaluation function
        self.gaussian = exp(self.Q)
        self.gaussian_lamb = lambdify(tuple((x,*varlist)), self.gaussian)
        self.dgaussian = []
        self.dgaussian_lamb = []
        # Save the derivatives of the gaussian with respect to the parameters
        for k in range(len(varlist)):
            self.dgaussian.append( simplify(diff( self.gaussian, varlist[k] )) )
            self.dgaussian_lamb.append( lambdify(tuple((x,*varlist)), self.dgaussian[k]) )

        if self.verbose:
            print('Gaussian and its derivatives:')
            display(Eq(Symbol('g'),self.gaussian))
            for k in range(len(varlist)):
                display(Eq(Symbol(f'g_{varlist[k].name}'),self.dgaussian[k] ) )


        # Solve for the Xi's in terms of the parameters

        eqn = collect(expand(self.Q-self.Q0),x)

        # Extract coefficients and solve for the alpha_i's to get the maps Phi_i
        # Save in a dict of funtions Phi_lamb.

        args = varlist

        if self.verbose:
            print('Coordinate maps: ')
        coeffs = Poly(eqn,x).all_coeffs()
        Phi = []
        self.Phi_lamb = []
        for i in range(len(coeffs)):
            coeff = coeffs[i]
            #display(coeff)
            X_i = Symbol(f'X_{i}')
            eq = solve(coeffs,X_i)[X_i]
            Phi.append(eq)
            if self.verbose:
                display(Eq(Symbol(f'Phi_{i}'), Phi[i]))

            #mydisplay(f'Phi[{i}]')
            self.Phi_lamb.append( lambdify(args,Phi[i]) )
            exec(f'self.Phi_lamb_{i} = lambdify(args,Phi[{i}])')
            #self.Phi_lamb_0 = lambdify(varlist,Phi[0])

        self.Phi = Phi

        # Compute Jacobian
        self.M = {}
        self.M_lamb0 = {}

        for i in range(len(Phi)):
            for j in range(len(args)):
                pdiff = diff(Phi[i],args[j])

                self.M[(i,j)] = pdiff
                self.M_lamb0[(i,j)] = lambdify(args,pdiff)
                if self.verbose:
                    display(Eq(Symbol(f'M_{i},{args[j].name}'),pdiff))

        def M_lamb(z):
            return np.array([self.M_lamb0[(i,j)](*z) for i in range(3) for j in range(len(args))]).reshape((3,len(args)))

        self.M_lamb = M_lamb
        

    def Phi(self,*args):
        X0 = self.Phi_lamb[0](*args)
        X1 = self.Phi_lamb[1](*args)
        X2 = self.Phi_lamb[2](*args)
        return np.array([X0,X1,X2])



class LCGauss1D:
    """ Class for computing with linear combinations of gaussians."""
    def __init__(self,g : Gauss1D = None):
        """ Constructor. Takes an instance of GaussIntegrals1D as parameter. 
        If none is given, the default set of gaussians is used."""
        if g is None:
            self.g = Gauss1D()
        else:
            self.g = g # instance of Gauss1D
        self.n_par = len(self.g.varlist)
        
        self.set_regularization(type_P = 'none', type_Q = 'none')

    def set_regularization(self, lambda_P = 0.0, type_P = 'none',lambda_Q = 0.0, type_Q = 'none'):
        """ Set regularization parameters for inverses of P and Q overlap matrices S and A."""
        
        if type_P == 'none':
            self.regularize_P = 'none'
        elif type_P == 'tikhonov':
            self.lambda_P = lambda_P
            self.regularize_P = 'tikhonov'
        elif type_P == 'smallsing':
            self.lambda_P = lambda_P
            self.regularize_P = 'smallsing'
        else:
            raise ValueError('Invalid regularization type for P matrix.')
        
        if type_Q == 'none':
            self.regularize_Q = 'none'
        elif type_Q == 'tikhonov':
            self.lambda_Q = lambda_Q
            self.regularize_Q = 'tikhonov'
        elif type_Q == 'smallsing':
            self.lambda_Q = lambda_Q
            self.regularize_Q = 'smallsing'
        else:
            raise ValueError('Invalid regularization type for Q matrix.')
        
            

    def set_potential(self, c, z):
        """ Set the potential. 
        
        The potential is defined as an LCG.
        
        Args:
            c: Coefficients of the LCG.
            z: Gauss parameters of the LCG.
        
        """
        assert(self.n_par == z.shape[0])
        self.c_pot = c # coefficients
        self.z_pot = z # gauss params
        self.n_pot = z.shape[1]
        
        self.X_pot = np.zeros((3,self.n_pot),dtype=complex)
        z_vec = tuple(z[m,:] for m in range(self.n_par))
        
        # Compute primitive parameters X from z.
        self.X_pot = np.array([self.g.Phi_lamb[i](*z_vec) for i in range(3)])
        #ic(self.X_pot)
        
    def set_z(self, z):
        """ Set the nonlinear coefficients. """
        
        assert(self.n_par == z.shape[0])
        
        self.z = z # gauss params
        self.n = z.shape[1]

        self.X = np.zeros((3,self.n),dtype=complex)
        z_vec = tuple(z[m,:] for m in range(self.n_par))
        
        # Compute primitive parameters X from z.
        self.X = np.array([self.g.Phi_lamb[i](*z_vec) for i in range(3)])
        
        # Create a tensor of all possible sums of the X and the conjugates.
        # Y[:,i,j] = X[:,i].conj() + X[:,j].
        # Very useful for vectorization of lambdifed functions.
        self.Y = self.X[:,:,np.newaxis].conj() + self.X[:,np.newaxis,:]
        
        # compute Jacobian of parameter transformation
        self.M = np.zeros((3,self.n_par, self.n),dtype=complex)
        for i in range(self.n):
                self.M[:,:,i] = self.g.M_lamb(self.z[:,i])
        #ic(self.M)

    def set_c(self, c):
        """ Set the coefficients. """
        self.c = c


        
        
    def overlap_matrices0(self):
        """ Compute overlap matrix of basis determined by z_vec. Also compute other overlaps needed. """
        n = self.n
        X = self.X
        
        #
        # The usual overlap matrix S = [<i|j>]
        # 

        # S = np.zeros((n,n),dtype=complex)
        # for i in range(n):
        #     for j in range(n):
        #         Y = X[:,i].conj() + X[:,j]
        #         S[i,j] = self.g.J(Y)

        
        # # Maybe faster?
        S_fast = self.g.J(self.Y)
        # self.S = S
        #assert(np.allclose(S, S_fast)) # why is the norm not identically zero?
        

        self.S = S_fast
        
        #
        # inverse of overlap matrix
        #
        
        if self.regularize_P == 'tikhonov':
            S_temp = self.S + self.lambda_P*np.eye(self.n)
        elif self.regularize_P == 'smallsing':
            S_temp = self.S + self.lambda_P*expm(-self.S/self.lambda_P)
        else:
            S_temp = self.S
            
        self.S_chol = cho_factor(S_temp) # S = L L^H
        self.S_inv = np.linalg.inv(S_temp)
            

        #
        # Overlaps F, F[m,i,j] = c_i.conj() * <partial_m i|j> 
        # 
        # F0 = np.zeros((3,n,n),dtype=complex)
        
        # for i in range(n):
        #     for j in range(n):
        #         Y = X[:,i].conj() + X[:,j]
        #         F0[0,i,j] = self.g.J100_lamb(Y)
        #         F0[1,i,j] = self.g.J010_lamb(Y)
        #         F0[2,i,j] = self.g.J001_lamb(Y)
        
        F0_fast = np.zeros((3,n,n),dtype=complex)
        F0_fast[0,:,:] = self.g.J100(self.Y)
        F0_fast[1,:,:] = self.g.J010(self.Y)
        F0_fast[2,:,:] = self.g.J001(self.Y)
        #assert(np.allclose(F0,F0_fast))
        
        F = np.einsum('nmi,nij->mij',self.M.conj(),F0_fast)
        F = np.einsum('mij,i->mij',F,self.c.conj())
        self.F = F

        #
        # Overlaps G, G[m,i,n,j] = <partial_m  i| partial_n j>
        #
        # compute the overlaps G[m,n,i,j] = <partial_m  i| partial_n j>.
        # G0 = np.zeros((3,3,n,n),dtype=complex)
        # for i in range(n):
        #     for j in range(n):
        #         Y = X[:,i].conj() + X[:,j]
        #         G0[0,0,i,j] = self.g.J200_lamb(Y)
        #         G0[0,1,i,j] = self.g.J110_lamb(Y)
        #         G0[0,2,i,j] = self.g.J101_lamb(Y)
        #         G0[1,0,i,j] = self.g.J110_lamb(Y)
        #         G0[1,1,i,j] = self.g.J020_lamb(Y)
        #         G0[1,2,i,j] = self.g.J011_lamb(Y)
        #         G0[2,0,i,j] = self.g.J101_lamb(Y)
        #         G0[2,1,i,j] = self.g.J011_lamb(Y)
        #         G0[2,2,i,j] = self.g.J002_lamb(Y)
                
        G0_fast = np.zeros((3,3,n,n),dtype=complex)
        G0_fast[0,0,:,:] = self.g.J200(self.Y)
        G0_fast[0,1,:,:] = self.g.J110(self.Y)
        G0_fast[0,2,:,:] = self.g.J101(self.Y)
        G0_fast[1,0,:,:] = self.g.J110(self.Y)
        G0_fast[1,1,:,:] = self.g.J020(self.Y)
        G0_fast[1,2,:,:] = self.g.J011(self.Y)
        G0_fast[2,0,:,:] = self.g.J101(self.Y)
        G0_fast[2,1,:,:] = self.g.J011(self.Y)
        G0_fast[2,2,:,:] = self.g.J002(self.Y)

 
#        assert(np.allclose(G0,G0_fast))
        
        G = np.einsum('nmi,opj,noij->mipj',self.M.conj(),self.M,G0_fast)
        G = np.einsum('mipj,i,j->mipj',G,self.c.conj(), self.c)
        self.G = G
        
                
        
        return S
        
        
    def overlap_matrix(self):
        """ Compute the standard overlap matrix only."""
        self.S = self.g.J(self.Y)
        
    def overlap_matrices(self):
        """ Compute overlap matrix of basis determined by z_vec. Also compute other overlaps needed. """
        n = self.n
        X = self.X
        
        #
        # The usual overlap matrix S = [<i|j>]
        # 

        # S = np.zeros((n,n),dtype=complex)
        # for i in range(n):
        #     for j in range(n):
        #         Y = X[:,i].conj() + X[:,j]
        #         S[i,j] = self.g.J_lamb(Y)

        
        # # Maybe faster?
        self.overlap_matrix()
        # self.S = S
        #assert(np.allclose(S, S_fast)) # why is the norm not identically zero?
        
        #
        # inverse of overlap matrix
        #
        
        if self.regularize_P == 'tikhonov':
            S_temp = self.S + self.lambda_P*np.eye(self.n)
        elif self.regularize_P == 'smallsing':
            S_temp = self.S + self.lambda_P*expm(-self.S/self.lambda_P)
        else:
            S_temp = self.S
            
        self.S_chol = cho_factor(S_temp) # S = L L^H
        self.S_inv = np.linalg.inv(S_temp)

        #
        # Overlaps F, F[m,i,j] = c_i.conj() * <partial_m i|j> 
        # 
        # F0 = np.zeros((3,n,n),dtype=complex)
        
        # for i in range(n):
        #     for j in range(n):
        #         Y = X[:,i].conj() + X[:,j]
        #         F0[0,i,j] = self.g.J100_lamb(Y)
        #         F0[1,i,j] = self.g.J010_lamb(Y)
        #         F0[2,i,j] = self.g.J001_lamb(Y)
        
        F0_fast = np.zeros((3,n,n),dtype=complex)
        F0_fast[0,:,:] = self.g.fJ100(self.Y) * self.S
        F0_fast[1,:,:] = self.g.fJ010(self.Y) * self.S
        F0_fast[2,:,:] = self.g.fJ001(self.Y) * self.S
        #assert(np.allclose(F0,F0_fast))
        
        F = np.einsum('nmi,nij->mij',self.M.conj(),F0_fast)
        F = np.einsum('mij,i->mij',F,self.c.conj())
        self.F = F

        #
        # Overlaps G, G[m,i,n,j] = <partial_m  i| partial_n j>
        #
        # compute the overlaps G[m,n,i,j] = <partial_m  i| partial_n j>.
        # G0 = np.zeros((3,3,n,n),dtype=complex)
        # for i in range(n):
        #     for j in range(n):
        #         Y = X[:,i].conj() + X[:,j]
        #         G0[0,0,i,j] = self.g.J200_lamb(Y)
        #         G0[0,1,i,j] = self.g.J110_lamb(Y)
        #         G0[0,2,i,j] = self.g.J101_lamb(Y)
        #         G0[1,0,i,j] = self.g.J110_lamb(Y)
        #         G0[1,1,i,j] = self.g.J020_lamb(Y)
        #         G0[1,2,i,j] = self.g.J011_lamb(Y)
        #         G0[2,0,i,j] = self.g.J101_lamb(Y)
        #         G0[2,1,i,j] = self.g.J011_lamb(Y)
        #         G0[2,2,i,j] = self.g.J002_lamb(Y)
                
        G0_fast = np.zeros((3,3,n,n),dtype=complex)
        G0_fast[0,0,:,:] = self.g.fJ200(self.Y) * self.S
        G0_fast[0,1,:,:] = self.g.fJ110(self.Y) * self.S
        G0_fast[0,2,:,:] = self.g.fJ101(self.Y) * self.S
        G0_fast[1,0,:,:] = self.g.fJ110(self.Y) * self.S
        G0_fast[1,1,:,:] = self.g.fJ020(self.Y) * self.S
        G0_fast[1,2,:,:] = self.g.fJ011(self.Y) * self.S
        G0_fast[2,0,:,:] = self.g.fJ101(self.Y) * self.S
        G0_fast[2,1,:,:] = self.g.fJ011(self.Y) * self.S
        G0_fast[2,2,:,:] = self.g.fJ002(self.Y) * self.S

 
#        assert(np.allclose(G0,G0_fast))
        
        G = np.einsum('nmi,opj,noij->mipj',self.M.conj(),self.M,G0_fast)
        G = np.einsum('mipj,i,j->mipj',G,self.c.conj(), self.c)
        self.G = G
        
                
        
        return S

    def hamiltonian_matrix(self):
        """ Compute Hamiltonian matrix. """

        n = self.n
        X = self.X
        
        fX_dip = -self.g.fJ010(self.Y) 

        A = X[1,:,np.newaxis].conj() * X[1,np.newaxis,:]
        B = X[1,:,np.newaxis].conj()*X[2,np.newaxis,:] + X[2,:,np.newaxis].conj()*X[1,np.newaxis,:]
        C = X[2,:,np.newaxis].conj()*X[2,np.newaxis,:]
        
        fT = 0.5*(A - 2*B*self.g.fJ010(self.Y) + 4*C*self.g.fJ020(self.Y))

        V = np.zeros((n,n),dtype=complex) # potential energy matrix
        for k in range(self.n_pot):
            Z = self.Y + self.X_pot[:,k,np.newaxis,np.newaxis]
            V += self.c_pot[k] * self.g.J(Z)

        self.H = fT * self.S + V
        self.X_dip = fX_dip * self.S
        
    def hamiltonian_matrix0(self):
        """ Compute Hamiltonian matrix. """

        n = self.n
        H = np.zeros((n,n),dtype=complex) # total Hamiltonian matrix
        X_dip = np.zeros((n,n),dtype=complex) # dipole matrix
        T = np.zeros((n,n),dtype=complex) # kinetic energy matrix
        V = np.zeros((n,n),dtype=complex) # potential energy matrix
        
        X = self.X
        
        #Y = X[:,:,np.newaxis].conj() + X[:,np.newaxis,:]
        
        #X_dip = -self.g.J110_lamb(Y)

        #temp1 = self.g.J010_lamb(Y)
        #temp2 = self.g.J020_lamb(Y)
        
        for i in range(n):
            for j in range(n):
                Y = X[:,i].conj() + X[:,j]
                # dipole matrix = <i|x|j>.
                X_dip[i,j] = -self.g.J010(Y) 
                
                # kinetic energy matrix <i'|j'>/2
                # = (1/2)(\bar{X}^i_1 X^j_1 J(Z) -2 (\bar{X}_1^i X_2^j + X_1^j\bar{X}_2^i) \partial_{1} S(Z) + 4 \bar{X}_2^i X_2^j\partial_1^2 S(Z))
                T[i,j] = 0.5*(X[1,i].conj()*X[1,j]*self.g.J(Y) - 2*(X[1,i].conj()*X[2,j] + X[2,i].conj()*X[1,j])*self.g.J010(Y) + 4*X[2,i].conj()*X[2,j]*self.g.J020(Y)) 
#                T[i,j] = 0.5*(X[1,i].conj()*X[1,j]*self.S[i,j] - 2*(X[1,i].conj()*X[2,j] + X[2,i].conj()*X[1,j])*temp1[i,j] + 4*X[2,i].conj()*X[2,j]*temp2[i,j])
                
                for k in range(self.n_pot):
                    Z = X[:,i].conj() + X[:,j] + self.X_pot[:,k]
                    V[i,j] += self.c_pot[k] * self.g.J(Z) 
                    
                #Z = Y + X[:,np.newaxis,:]
        # for k in range(self.n_pot):
        #     Z = Y + self.X_pot[:,k,np.newaxis,np.newaxis]
        #     V += self.c_pot[k] * self.g.J_lamb(Z)

        self.V = V
        self.H = T + V
        self.X_dip = X_dip
        
        
    def mclachlan_ode0(self, z_in, c_in, E_field = 0.0):
        """ Compute the McLachlan VP ODE."""

        # Start with computing the overlap matrix and
        # Hamiltonian matrix.
        self.set_z(z_in)
        self.set_c(c_in)
        self.overlap_matrices0()
        self.hamiltonian_matrix0()
        
        # Compute the overlaps F[m,i,j] = <partial_m i|j>.

        F = self.F

        G = self.G
        
        n = self.n
        n_par = self.n_par
        X = self.X
        
        #ic(G.shape)
        
        # compute the Q-space overlap matrix
        # A[m,i,n,j] = <partial_m i|partial_n j> - <partial_m i|k>S_inv(k,l)<l|partial_n j><partial_m i|j>.

#        F_temp = F.conj().transpose(2,1,0).reshape((n,n_par*n))
#        temp = cho_solve(self.S_chol, F_temp).reshape((n,n_par,n)).transpose(1,0,2)
#        ic(temp.shape, F.shape)
#        A0 = G - np.einsum('mil,njl->minj', F, temp)
        A0 = G - np.einsum('mik,kl,njl->minj', F, self.S_inv, F.conj())
        #ic(A.shape)
        A = A0.real.reshape((n_par*n,n_par*n))
        self.A = A
        #ic(np.linalg.norm(A - A.T))
        #ic(np.linalg.cond(A))
        #ic(A2.shape)


        # compute the Q-space vector field
        # b = Im[<partial m i|H|j>c_j - <partial m i|>S_inv(i,l)<l|H|j> c_j]
        # first compute u = <partial m i|H|j>
        u = np.zeros((n_par,n,n),dtype=complex)
        u0 = np.zeros((3,n,n),dtype=complex)
        
        
        for i in range(n):
            for j in range(n):

                Y = X[:,i].conj() + X[:,j]

         
                u0[0,i,j] = 0.5*(X[1,i].conj()*X[1,j]*self.g.J100(Y) - 2*(X[1,i].conj()*X[2,j] + X[2,i].conj()*X[1,j])*self.g.J110(Y) + 4*X[2,i].conj()*X[2,j]*self.g.J120(Y))
                u0[1,i,j] = 0.5*(X[1,i].conj()*X[1,j]*self.g.J010(Y) - 2*(X[1,i].conj()*X[2,j] + X[2,i].conj()*X[1,j])*self.g.J020(Y) + 4*X[2,i].conj()*X[2,j]*self.g.J030(Y)) + 0.5*(X[1,j]*self.g.J(Y) - 2*X[2,j]*self.g.J010(Y))
                u0[2,i,j] = 0.5*(X[1,i].conj()*X[1,j]*self.g.J001(Y) - 2*(X[1,i].conj()*X[2,j] + X[2,i].conj()*X[1,j])*self.g.J011(Y) + 4*X[2,i].conj()*X[2,j]*self.g.J021(Y)) - X[1,j]*self.g.J010(Y) + 2*X[2,j]*self.g.J020(Y)
                
                #
                # Potential energy term
                #
                for k in range(self.n_pot):
                    Z = Y + self.X_pot[:,k]
                    u0[0,i,j] += self.c_pot[k] * self.g.J100(Z)
                    u0[1,i,j] += self.c_pot[k] * self.g.J010(Z)
                    u0[2,i,j] += self.c_pot[k] * self.g.J001(Z)
                    
                #
                # dipole term
                #
                u0[0,i,j] += -E_field * self.g.J110(Y)
                u0[1,i,j] += -E_field * self.g.J020(Y)
                u0[2,i,j] += -E_field * self.g.J011(Y)  
        
        
        u0 = np.einsum('nij,i->nij',u0, self.c.conj())
        u = np.einsum('nmi,nij->mij',self.M.conj(),u0)
        temp = cho_solve(self.S_chol, (self.H + self.X_dip*E_field))
        u = u - np.einsum('mik,kj->mij', F, temp)
        v0 = np.einsum('mij,j->mi',u,c_in)           
        v = v0.imag.reshape((n_par*n,))
        self.v = v
        #ic(v2.shape)
        
        #
        # We now have everything ...
        #
        dzdt = np.linalg.solve(A,v).reshape((n_par,n))
            
        # coriolis term
        # <i|partial m j> dzdt[m,j] c_j
        S_dcdt = -1j * (self.H + self.X_dip*E_field) @ c_in - np.einsum('mji,mj->i',F.conj(),dzdt)
        #ic(S_dcdt.shape)
        dcdt = self.S_inv @ S_dcdt
        return dzdt, dcdt        
 
    def mclachlan_ode(self, z_in, c_in, E_field = 0.0):
        """ Compute the McLachlan VP ODE."""

        # Start with computing the overlap matrix and
        # Hamiltonian matrix.
        self.set_z(z_in)
        self.set_c(c_in)
        self.overlap_matrices()
        self.hamiltonian_matrix()

        F = self.F
        G = self.G
        
        n = self.n
        n_par = self.n_par
        X = self.X
        
        A0 = G - np.einsum('mik,kl,njl->minj', F, self.S_inv, F.conj())
        A = A0.real.reshape((n_par*n,n_par*n))
        self.A = A


        # compute the Q-space vector field
        # b = Im[<partial m i|H|j>c_j - <partial m i|>S_inv(i,l)<l|H|j> c_j]
        # first compute u = <partial m i|H|j>
        u = np.zeros((n_par,n,n),dtype=complex)
        u0 = np.zeros((3,n,n),dtype=complex)
        
        A1 = X[1,:,np.newaxis].conj()*X[1,np.newaxis,:]
        A2 = X[1,:,np.newaxis].conj()*X[2,np.newaxis,:] + X[2,:,np.newaxis].conj()*X[1,np.newaxis,:]
        A3 = X[2,:,np.newaxis].conj()*X[2,np.newaxis,:]
        u0[0,:,:] = 0.5*(A1*self.g.fJ100(self.Y) - 2*A2*self.g.fJ110(self.Y) + 4*A3*self.g.fJ120(self.Y))
        u0[1,:,:] = 0.5*(A1*self.g.fJ010(self.Y) - 2*A2*self.g.fJ020(self.Y) + 4*A3*self.g.fJ030(self.Y))
        u0[2,:,:] = 0.5*(A1*self.g.fJ001(self.Y) - 2*A2*self.g.fJ011(self.Y) + 4*A3*self.g.fJ021(self.Y))
        u0[1,:,:] += 0.5*(X[1,np.newaxis,:] - 2*X[2,np.newaxis,:]*self.g.fJ010(self.Y))
        u0[2,:,:] += -X[1,np.newaxis,:]*self.g.fJ010(self.Y) + 2*X[2,np.newaxis,:]*self.g.fJ020(self.Y)
 
        u0[0,:,:] += -E_field * self.g.fJ110(self.Y)
        u0[1,:,:] += -E_field * self.g.fJ020(self.Y)
        u0[2,:,:] += -E_field * self.g.fJ011(self.Y)         

        u0 *= self.S[np.newaxis,:,:]

        # potential terms cannot be factorized ... 
        for k in range(self.n_pot):
            Z = self.Y + self.X_pot[:,np.newaxis,np.newaxis,k]
            u0[0,:,:] += self.c_pot[k] * self.g.J100(Z)
            u0[1,:,:] += self.c_pot[k] * self.g.J010(Z)
            u0[2,:,:] += self.c_pot[k] * self.g.J001(Z)       
        
        
        u0 = np.einsum('nij,i->nij',u0, self.c.conj())
        u = np.einsum('nmi,nij->mij',self.M.conj(),u0)
        temp = cho_solve(self.S_chol, (self.H + self.X_dip*E_field))
        u = u - np.einsum('mik,kj->mij', F, temp)
        v0 = np.einsum('mij,j->mi',u,c_in)           
        v = v0.imag.reshape((n_par*n,))
        self.v = v
        
        #
        # We now have everything ...
        #
        #
        # We now have everything ...
        #
        if self.regularize_Q == 'tikhonov':
            A_temp = A + np.eye(A.shape[0])*self.lambda_Q
        elif self.regularize_Q == 'smallsing':
            A_temp = A + self.lambda_Q*expm(-A/self.lambda_Q)
        else:
            A_temp = A
            
        dzdt = np.linalg.solve(A_temp,v).reshape((n_par,n))

        # coriolis term
        # <i|partial m j> dzdt[m,j] c_j
        S_dcdt = -1j * (self.H + self.X_dip*E_field) @ c_in - np.einsum('mji,mj->i',F.conj(),dzdt)
        #ic(S_dcdt.shape)
        dcdt = self.S_inv @ S_dcdt
        return dzdt, dcdt        
        
    def mclachlan_ode_imag_time(self, z_in, c_in):
        """ Compute the McLachlan VP ODE for imag time prop."""

        # Start with computing the overlap matrix and
        # Hamiltonian matrix.
        self.set_z(z_in)
        self.set_c(c_in)
        self.overlap_matrices()
        self.hamiltonian_matrix()

        F = self.F
        G = self.G
        
        n = self.n
        n_par = self.n_par
        X = self.X
        
        A0 = G - np.einsum('mik,kl,njl->minj', F, self.S_inv, F.conj())
        A = A0.real.reshape((n_par*n,n_par*n))
        self.A = A


        # compute the Q-space vector field
        # b = Im[<partial m i|H|j>c_j - <partial m i|>S_inv(i,l)<l|H|j> c_j]
        # first compute u = <partial m i|H|j>
        u = np.zeros((n_par,n,n),dtype=complex)
        u0 = np.zeros((3,n,n),dtype=complex)
        
        A1 = X[1,:,np.newaxis].conj()*X[1,np.newaxis,:]
        A2 = X[1,:,np.newaxis].conj()*X[2,np.newaxis,:] + X[2,:,np.newaxis].conj()*X[1,np.newaxis,:]
        A3 = X[2,:,np.newaxis].conj()*X[2,np.newaxis,:]
        u0[0,:,:] = 0.5*(A1*self.g.fJ100(self.Y) - 2*A2*self.g.fJ110(self.Y) + 4*A3*self.g.fJ120(self.Y))
        u0[1,:,:] = 0.5*(A1*self.g.fJ010(self.Y) - 2*A2*self.g.fJ020(self.Y) + 4*A3*self.g.fJ030(self.Y))
        u0[2,:,:] = 0.5*(A1*self.g.fJ001(self.Y) - 2*A2*self.g.fJ011(self.Y) + 4*A3*self.g.fJ021(self.Y))
        u0[1,:,:] += 0.5*(X[1,np.newaxis,:] - 2*X[2,np.newaxis,:]*self.g.fJ010(self.Y))
        u0[2,:,:] += -X[1,np.newaxis,:]*self.g.fJ010(self.Y) + 2*X[2,np.newaxis,:]*self.g.fJ020(self.Y)
 

        u0 *= self.S[np.newaxis,:,:]

        # potential terms cannot be factorized ... 
        for k in range(self.n_pot):
            Z = self.Y + self.X_pot[:,np.newaxis,np.newaxis,k]
            u0[0,:,:] += self.c_pot[k] * self.g.J100(Z)
            u0[1,:,:] += self.c_pot[k] * self.g.J010(Z)
            u0[2,:,:] += self.c_pot[k] * self.g.J001(Z)       
        
        
        u0 = np.einsum('nij,i->nij',u0, self.c.conj())
        u = np.einsum('nmi,nij->mij',self.M.conj(),u0)
        temp = cho_solve(self.S_chol, self.H)
        u = u - np.einsum('mik,kj->mij', F, temp)
        v0 = np.einsum('mij,j->mi',u,c_in)           
        v = -v0.real.reshape((n_par*n,)) # for imag time prop, neg real part instead of imag part
        self.v = v
        
        #
        # We now have everything ...
        #
        N2 = (c_in.conj() @ self.S @ c_in).real
        E = np.sum(c_in.conj() @ self.H @ c_in) / N2

        #
        # We now have everything ...
        #
        if self.regularize_Q == 'tikhonov':
            A_temp = A + np.eye(A.shape[0])*self.lambda_Q
        elif self.regularize_Q == 'smallsing':
            A_temp = A + self.lambda_Q*expm(-A/self.lambda_Q)
        else:
            A_temp = A
            
        dzdt = np.linalg.solve(A_temp,v).reshape((n_par,n))
        
        
        dzdt /= N2
        # coriolis term
        # <i|partial m j> dzdt[m,j] c_j
        
        # for imag time prop -1j goes to -1. Also renormalization term
        S_dcdt = (- self.H @ c_in + E * self.S @ c_in - np.einsum('mji,mj->i',F.conj(),dzdt))/N2
        #ic(S_dcdt.shape)
        dcdt = self.S_inv @ S_dcdt
        return dzdt, dcdt        

    def par_to_vec(self, c, z):
        """ Convert (c, z) to a real vector """
        n = len(c)
        n_par = self.n_par
        
        y = np.zeros((n*n_par + 2*n,))
        y[:n] = c.real
        y[n:2*n] = c.imag
        y[2*n:] = z.reshape(-1)

        return y            
            
    def vec_to_par(self, y):
        """ Convert vector representation to complex c vector and real z matrix """
        n = len(y) // (self.n_par + 2)
        n_par = self.n_par
        assert len(y) == n*n_par + 2*n
        return  y[:n] + y[n:2*n] * 1j, y[2*n:].reshape((n_par, n))


    def get_odefun0(self, laser):
        
        def odefun(t,y):
            c, z = self.vec_to_par(y)
            dzdt, dcdt = self.mclachlan_ode0(z,c,E_field=laser(t))
            return self.par_to_vec(dcdt, dzdt)

        return odefun        
        

    def get_odefun(self, laser):
        
        def odefun(t,y):
            c, z = self.vec_to_par(y)
            dzdt, dcdt = self.mclachlan_ode(z,c,E_field=laser(t))
            return self.par_to_vec(dcdt, dzdt)

        return odefun        
        
    def get_odefun_imag_time(self):
        
        def odefun(t,y):
            c, z = self.vec_to_par(y)
            dzdt, dcdt = self.mclachlan_ode_imag_time(z,c)
            return self.par_to_vec(dcdt, dzdt)

        return odefun        
                
    def eval_grid(self,x, c=None):
        """ Evaluate wavefunction on a grid. """

        if c is None:
            c = self.c
            
        gauss = self.g.gaussian_lamb

        ans = np.zeros(x.shape, dtype=complex)
        for k in range(len(c)):
            ans += gauss(x,*tuple(self.z[:,k])) * c[k]

        return ans

    def projection(self,x,psi):
        """ Project wavefunction onto current basis defined by z_vec. Return c. """

        gauss = self.g.gaussian_lamb

        g = np.zeros(self.n, dtype=complex)

        for k in range(self.n):
            g[k] = np.sum(gauss(x,*self.z[:,k]).conj() * psi) * (x[1]-x[0])

        S = self.g.J(self.Y)
        return np.linalg.solve(S, g)




def test1():
    #
    # Do a test of the speeup of vectorization.
    #
    
    from time import time

    g = Gauss1D(verbose=True)

    # Set up a test case
    n = 500
    np.random.seed(0)
    X = np.random.rand(3,n) + 1j*np.random.rand(3,n)
    Y = X[:,:,np.newaxis].conj() + X[:,np.newaxis,:]
    
    # Compute the overlap matrix in non-vectorized manner
    tic = time()
    S = np.zeros((n,n),dtype=complex)
    for i in range(n):
        for j in range(n):
            Y1 = X[:,i].conj() + X[:,j]
            S[i,j] = g.J(Y1)
    toc = time()
    time_overlap_non_vectorized = toc-tic
    ic(time_overlap_non_vectorized)


    # Compute the overlap matrix in vectorized manner           
    tic = time()
    S = g.J(Y)
    toc = time()
    time_overlap_vectorized = toc-tic
    ic(time_overlap_vectorized)
    ic(time_overlap_non_vectorized/time_overlap_vectorized)
    ic(S[0,0])
    S_inv = np.linalg.inv(S)
    
    # Compute a derivative in non-vectorized manner
    tic = time()
    J = np.zeros((n,n),dtype=complex)
    for i in range(n):
        for j in range(n):
            Y1 = X[:,i].conj() + X[:,j]
            J[i,j] = g.J110(Y1)
    toc = time()
    time_derivative_non_vectorized = toc-tic
    
    # Compute a derivative in vectorized manner    
    tic = time()
    J = g.J110(Y)
    toc = time()
    time_derivative_vectorized = toc-tic
    ic(time_derivative_non_vectorized)
    ic(time_derivative_vectorized)
    ic(time_derivative_non_vectorized/time_derivative_vectorized)
    
    # Compute a derivative in a vectorized and factorized manner
    tic = time()
    J2 = g.fJ110(Y) * S 
    toc = time()
    time_derivative_vectorized_factorized = toc-tic
    ic(time_derivative_vectorized_factorized)
    ic(time_derivative_vectorized/time_derivative_vectorized_factorized)
    ic(time_derivative_non_vectorized/time_derivative_vectorized_factorized)
    ic(np.allclose(J,J2))

    
def test2():
        
    from time import time
    import pickle
    
    g = Gauss1D(verbose=True)
    df = LCGauss1D(g)

    # load initial state from pickle
    fname = 'lcg_optimized_ngauss_10.pkl'
    with open(fname, 'rb') as f:
        initial_state = pickle.load(f)
        c = initial_state['c']
        z = initial_state['z']
        #z = z[:,:2]
        #c = c[:2]
        n = len(c)
                
    df.set_z(z)
    df.set_c(c)
    
    mu = 0.1
    z_pot = np.array([np.log(2*mu), 0.0, 1.3, 0.0]).reshape(4,1)
    c_pot = np.array([-1.0])
    df.set_potential(c_pot, z_pot)
    
    tic = time()
    df.overlap_matrices0()
    toc = time()
    time_overlap0 = toc-tic
    ic(toc-tic)

    S0 = df.S
    F0 = df.F
    G0 = df.G

    tic = time()
    df.overlap_matrices()
    toc = time()
    time_overlap = toc-tic
    ic(toc-tic)
    ic(time_overlap0/time_overlap)

    S = df.S
    F = df.F
    G = df.G
    
    ic(np.linalg.norm(S0-S))
    ic(np.linalg.norm(F0-F))
    ic(np.linalg.norm(G0-G))
    ic(np.linalg.norm(S0))
    ic(np.linalg.norm(F0))
    ic(np.linalg.norm(G0))
    

    tic = time()
    df.hamiltonian_matrix0()
    toc = time()
    time_hamiotonian0 = toc-tic
    ic(toc-tic)

    H0 = df.H.copy()
    X_dip0 = df.X_dip.copy()
    
        
    tic = time()
    df.hamiltonian_matrix()
    toc = time()
    time_hamiotonian = toc-tic
    ic(toc-tic)
    ic(time_hamiotonian0/time_hamiotonian)
    
    H = df.H.copy()
    X_dip = df.X_dip.copy()
    
    ic(np.linalg.norm(H0-H))
    ic(np.linalg.norm(X_dip0-X_dip))


    
    odefun0 = df.get_odefun0(lambda t: 0.0)
    odefun = df.get_odefun(lambda t: 0.0)
    y0 = df.par_to_vec(c,z)
    
    tic = time()
    dydt0 = odefun0(0,y0)
    toc = time()
    time_ode0 = toc-tic
    ic(toc-tic)
    
    A0 = df.A
    v0 = df.v
    
    tic = time()
    dydt = odefun(0,y0)
    toc = time()
    time_ode = toc-tic
    ic(toc-tic)
    
    A = df.A
    v = df.v
    
    ic(time_ode0/time_ode)
    ic(np.linalg.norm(dydt-dydt0))
    
    # S has large condition number --> we get errors in A, even if
    # all intermediates are machine precision close in the two cases
    ic(np.linalg.cond(df.S))
    ic(np.linalg.norm(A-A0))
    ic(np.linalg.norm(v-v0))
    

    
    
if __name__ == "__main__":
    
    test2()
    