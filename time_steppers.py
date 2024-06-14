import numpy as np
from scipy.special import gammaln
from icecream import ic

exp = np.exp

def crank_nicolson_step(y, t, dt, A_fun):
    """ Simple implementation of the Crank-Nicolson method for ODEs.
    
    $$ 
    y'(t) = -i A(t) y(t)
    $$
    
    Args:
    y: current state
    t: current time
    dt: time step
    A_fun: function that returns the matrix A at time t.
    """
    # Crank-Nicolson with CMF approximation
    A_mid = A_fun(t + dt/2)
    y_tmp = y - 1j * dt / 2 * np.dot(A_mid, y)
    y_new = np.linalg.solve(np.eye(len(y)) + 1j * dt / 2 * A_mid, y_tmp)
    return y_new

def rk4_step(t, y, dt, f):
    """Simple implementation of a Runge-Kutta 4th order ODE solver"""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def rk3_step(t, y, dt, f):
    """Simple implementation of a Kutta's 3rd order method"""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
    return y + (k1 + 4 * k2 + k3) * dt / 6


def heun_step(t, y, dt, f):
    """Simple implementation of Heun's method, a second-order explicit Runge-Kutta method"""
    k1 = f(t, y)
    k2 = f(t + dt, y + k1 * dt)
    return y + 0.5 * (k1 + k2) * dt


def ralston_step(t, y, dt, f):
    """Ralston's method is a second-order method, similar to Heun's method"""
    k1 = f(t, y)
    k2 = f(t + 2 / 3 * dt, y + 2 / 3 * k1 * dt)
    return y + 0.25 * k1 * dt + 0.75 * k2 * dt


def midpoint_step(t, y, dt, f):
    """Midpoint method is a second-order method"""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + k1 * dt / 2)
    return y + k2 * dt

def ode_step(y, t, dt, f, method=rk4_step):
    """Wrapper around the various ODE solvers."""
    return method(y, t, dt, f)


def compute_phi_0(H, dt):
    """Compute phi_0(A) = U phi_0(D) U^H where A = U D U^H."""
    E, U = np.linalg.eigh(H)
    phi_0 = np.exp(-1j * dt * E)
    return U @ np.diag(phi_0) @ U.T.conj()


def compute_phi_1(H, dt):
    """Compute phi_1(A) = U phi_1(D) U^H where A = U D U^H."""
    E, U = np.linalg.eigh(H)
    phi_10 = lambda z: (np.exp(z) - 1) / z if z != 0 else 1
    phi_1 = np.vectorize(phi_10)

    ans = U @ np.diag(phi_1(-1j * dt * E)) @ U.T.conj()

    return ans


def exp_euler(y, t, dt, f, phis):
    # Exponential Euler for the ODE y' + Ay = f(y,t)
    # Args:
    # y: current state
    # t: current time
    # dt: time step
    # f: function that returns the nonlinear/nonstiff part  of the ODE
    # phis: a dict of phi precomputed functions, phi[(i,j)] = phi_i(- h*A/j)
    # assumed precomputed

    phi_0 = phis[(0, 1)]
    phi_1 = phis[(1, 1)]
    #ic(y)
    #ic(phi_0 @ y)
    y_new = (phi_0 @ y) + dt * phi_1 @ f(y, t)
    return y_new


def exp_midpoint(y, t, dt, f, phis):
    phi_0 = phis[(0, 1)]
    phi_1 = phis[(1, 1)]
    phi_1_2 = phis[(1, 2)]
    phi_0_2 = phis[(0, 2)]
    y1 = phi_0_2 @ y + (dt / 2) * phi_1_2 @ f(y, t)
    y_new = phi_0 @ y + dt * phi_1 @ f(y1, t + dt / 2)
    return y_new


# def exp_imp_midpoint(y, t, dt, f, phis):
#     phi_0 = phis[(0, 1)]
#     phi_1 = phis[(1, 1)]
#     phi_1_2 = phis[(1, 2)]
#     phi_0_2 = phis[(0, 2)]
#     y1 = phi_0_2 @ y + (dt / 2) * phi_1_2 @ f(y, t)
#     y_new = phi_0 @ y + dt * phi_1 @ f(y1, t + dt / 2)
#     return y_new


def phi_func0(i, z):
    """Compute phi_i(z) according to the definition
    in the paper by Hochbruck, Ostermann, Exponential Integrators, Acta Numerica, 2010.

    A very naive implementation, but is should be all right
    for our purposes.
    

    
    """

    assert i >= 0
    if i == 0:
        return np.exp(z)
    else:
        return (
            (phi_func0(i - 1, z) - phi_func0(i - 1, 0)) / z
            if z != 0
            else exp(-gammaln(i + 1))
        )

phi_func = np.vectorize(phi_func0)

class ExponentialRungeKutta:
    """A simple class implementation of explicit
    exponential RK methods for the ode
    y' = -iA*y + f(y,t).

    The Butcher tableau looks like this:
    c1 |                             | chi1
    c2 | a21                         | chi2
    c3 | a31 a32                     | chi3
    ...|    ...                      | ...
    cs | as1 as2 ... as(s-1)         | chis
    ----------------------------------------------
            b1  b2 ...  bs              | chi
            
    chi_i = exp(-c[i]*h*A) for explicit methods.
    chi = exp(-h*A).
    
    """

    def __init__(self, A, f, dt, method="ExpEuler"):

        self.A = A
        # we do an explicit diagonalization in order
        # to compute the phi functions. this should
        # be made more general in a "real" implementation
        self.E, self.U = np.linalg.eigh(self.A)
        self.f = f
        self.dt = dt
        self.phi = {}
        self.c = None
        self.b = None
        self.a = None
        self.chi = None

        # Set up the default method
        self.prepare(method)


    def to_U(self,D):
        """Create a matrix diagonal in self.U."""
        return self.U @ np.diag(D) @ self.U.T.conj()

    def init_stage_s_method(self, s):
        """ Initialize the Butcher tableau for a method with s stages,
        empty data."""
        self.s = s
        self.c = np.zeros((s,), dtype=float)
        self.b = [None]*self.s
        self.chi = [None]*self.s
        self.a = [[None]*self.s]*self.s
        
        
        
    def prepare(self, method=None):

        if method is not None:
            self.method = method

        match self.method:
            case "ExpEuler":
                self.init_stage_s_method(1)
                # the c vector is only one element for the explicit Euler method
                self.c[0] = 0

                # the a matrix is empty for the explicit Euler method
                #...
                
                # set up chi functions
                # index 0 the lower right corner of the Butcher tableau
                # thus self.chi[i] = exp(-c[i-1]*h*A) ...
                #self.chi = [None]*(self.s)
                self.chi = [self.to_U(phi_func(0, -1j*self.E * self.dt * self.c[i])) for i in range(self.s)]
                self.chi0 = self.to_U(phi_func(0, -1j*self.E * self.dt))
                
                # set up b vector
                self.b[0] = self.to_U(phi_func(1, -1j*self.E * self.dt))
            case 'StrehmelWeider2':
                # The method of Strehmel and Weider, 1992, Example 4.2.2.
                # This is a second order method.
                c2 = 0.5
                self.init_stage_s_method(2)
                self.c[0] = 0
                self.c[1] = c2
                self.a[1][0] = c2 * self.to_U(phi_func(1, -1j*self.E * self.dt * c2))
                self.chi = [self.to_U(phi_func(0, -1j*self.E * self.dt * self.c[i])) for i in range(self.s)]
                self.chi0 = self.to_U(phi_func(0, -1j*self.E * self.dt))
                self.b[0] = self.to_U(phi_func(1, -1j*self.E * self.dt)) - self.to_U(phi_func(2, -1j*self.E * self.dt))/(2*c2) 
                self.b[1] = self.to_U(phi_func(2, -1j*self.E * self.dt))/(2*c2)
                
                
            case 'CoxMatthews4':
                # Example 2.19 in Hochbruck, Ostermann, Exponential Integrators, Acta Numerica, 2010.
                self.init_stage_s_method(4)
                self.c = np.array([0, 1/2, 1/2, 1])
                phi12 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[1]))
                phi13 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[2]))
                phi03 = self.to_U(phi_func(0, -1j*self.E * self.dt * self.c[2]))
                #phi14 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[3]))
                #phi23 = self.to_U(phi_func(2, -1j*self.E * self.dt * self.c[2]))
                #phi24 = self.to_U(phi_func(2, -1j*self.E * self.dt * self.c[3]))
                phi1 = self.to_U(phi_func(1, -1j*self.E * self.dt))
                phi2 = self.to_U(phi_func(2, -1j*self.E * self.dt))
                phi3 = self.to_U(phi_func(3, -1j*self.E * self.dt))
                
                self.a[1][0] = .5 * phi12
                self.a[2][1] = .5 * phi13
                self.a[3][2] = phi13
                self.a[3][0] = -.5 * phi13 @ (np.eye(len(self.E)) - phi03)
                self.chi = [self.to_U(phi_func(0, -1j*self.E * self.dt * self.c[i])) for i in range(self.s)]
                self.chi0 = self.to_U(phi_func(0, -1j*self.E * self.dt))
                
                self.b[0] = phi1 - 3*phi2 + 4*phi3
                self.b[1] = 2*phi2 - 4*phi3
                self.b[2] = 2*phi2 - 4*phi3
                self.b[3] = 4*phi3 - phi2
                
            case 'HochbruckOstermann4':
                # Example 2.19 in Hochbruck, Ostermann, Exponential Integrators, Acta Numerica, 2010.
                # Order 4 method with 5 stages.
                self.init_stage_s_method(5)
                self.c = np.array([0, 1/2, 1/2, 1, 1/2])
                phi12 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[1]))
                phi13 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[2]))
                phi15 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[4]))
                phi25 = self.to_U(phi_func(2, -1j*self.E * self.dt * self.c[4]))
                phi35 = self.to_U(phi_func(3, -1j*self.E * self.dt * self.c[4]))
                phi03 = self.to_U(phi_func(0, -1j*self.E * self.dt * self.c[2]))
                phi14 = self.to_U(phi_func(1, -1j*self.E * self.dt * self.c[3]))
                phi23 = self.to_U(phi_func(2, -1j*self.E * self.dt * self.c[2]))
                phi24 = self.to_U(phi_func(2, -1j*self.E * self.dt * self.c[3]))
                phi34 = self.to_U(phi_func(3, -1j*self.E * self.dt * self.c[3]))
                phi1 = self.to_U(phi_func(1, -1j*self.E * self.dt))
                phi2 = self.to_U(phi_func(2, -1j*self.E * self.dt))
                phi3 = self.to_U(phi_func(3, -1j*self.E * self.dt))
                
                self.a[1][0] = .5 * phi12
                self.a[2][0] = 0.5 * phi13 - phi23
                self.a[2][1] = phi23
                self.a[3][0] = phi14 - 2*phi24
                self.a[3][1] = phi24
                self.a[3][2] = phi24
                a52 = 0.5 * phi25 - phi34 + 0.25*phi24 - 0.5*phi35
                a54 = .25 * phi25 - a52
                self.a[4][0] = .5 * phi15 - 2*a52 - a54
                self.a[4][1] = a52
                self.a[4][2] = a52
                self.a[4][3] = a54

                self.chi = [self.to_U(phi_func(0, -1j*self.E * self.dt * self.c[i])) for i in range(self.s)]
                self.chi0 = self.to_U(phi_func(0, -1j*self.E * self.dt))
                
                self.b[0] = phi1 - 3*phi2 + 4*phi3
                self.b[3] = -phi2 + 4*phi3
                self.b[4] = 4*phi2 - 8*phi3
                
            case _:
                raise ValueError(f"Method '{method}' not implemented")

        #ic(self.method, self.dt)


    def step(self, y, t):
        """Generic explicit exponential Runge-Kutta method
        for the nonlinear ODE y' + Ay = f(y,t).

        Args:
        y: current state
        t: current time
        f: function that returns the nonlinear/nonstiff part  of the ODE

        Ref: Hochbruck, Ostermann, Exponential Integrators, Acta Numerica, 2010.
        See page 211 for the Butcher tableau.

        Returns:
        y_new: new state
        """

        s = len(self.c)
        U = np.zeros((s, *y.shape), dtype=np.complex128)
        G = np.zeros((s, *y.shape), dtype=np.complex128)
        for i in range(s):
            U[i] = self.chi[i] @ y
            for j in range(i):
                if self.a[i][j] is not None:
                    
                    U[i] += self.dt * self.a[i][j] @ G[j]

            G[i] = self.f(U[i], t + self.c[i] * self.dt)

        #ic(y)
        #ic(self.chi0 @ y)
        
        y_new = self.chi0 @ y
        for i in range(s):
            if self.b[i] is not None:
                y_new += self.dt * self.b[i] @ G[i]









        return y_new
