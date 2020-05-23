#==========================================================
from ..lib._matrix cimport matrix, vector

from ..lib._site cimport site
from ..lib._site cimport link_index as index

import numpy as np
import itertools
import time

exp = np.exp

#==========================================================
# Copy this block to any new application similar to those
# in `gauge_tools/util/` in order to set up the global variables.

def set_param(param):
    global nx, ny, nz, nt, dim, nc
    nx, ny, nz, nt, dim, nc = param.nx, param.ny, param.nz, param.nt, param.dim, param.nc
    global num_sites, num_links
    num_sites = param.num_sites
    num_links = param.num_links
    global ALL_links, ALL_sites
    ALL_sites = param.ALL_sites
    ALL_links = param.ALL_links

cdef int nx, ny, nz, nt
cdef int dim
cdef int nc

cdef int num_sites, num_links

ALL_sites = lambda: []
ALL_links = lambda: []

#==========================================================
class quark_field(object):
    """ A class to define quarks and calculate quark propagators.
    One can use the methods of this class to specify the source type
    and also calculate quark propagators.
    """
    def __init__(self, param):
        """ The input is an instance of :class:`gauge_tools.param`,
        which will be used to set the global parameters.
        """
        set_param(param)
    #===================================
    def set_source(self, src_type, t0=0, color_list=[0], **src_kwargs):
        """ sets the source. Parameters specify the source. """
        self.src = quark_src(src_type,t0,**src_kwargs)
        self.src.build_src(color_list)
    #===================================
    def calc_propagator(self, U, mass, n_restart=1, tadpole=False):
        """ calculates the quark propagators assuming the source is already defined.

        Parameters:
            - ``U``      (numpy.ndarray): the gauge links.
            - ``mass``   (float):   the mass of the quark.
            - ``n_restart`` (int):  number of restarts in calculation of quark\
                                    propagator (to account for numerical errors).
            - ``tadpole``  (bool):  if set `True`, the links will be tadpole improved.
                
        """
        if tadpole:
            import gauge_tools as gt
            u0 = gt.gauge_field().avg_links(U).trace().real/3.
        else:
            u0 = 1
        self.src_precond = self.src.evenodd_preconditioner(U, mass, u0=u0)
        self.solver = [cg_solver() for parity in [0,1]]
        self.prop = propagator(self.src_precond)
        for i in range(n_restart):
            print("\n>>> calculating quark propagator using conjugate gradient method (n_restart=%i)"%i)
            if tadpole: print("Links are tadpole improved by `average` of links set to %g"%u0)
            for parity in [0,1]:
                b = self.src_precond.get_v_src(parity)
                A = self.ks_MdM_operator(U, mass, parity, u0_sq=u0**2)
                x = self.prop.get_v_field(parity) if i>0 else None
                self.prop.set_v_field( self.solver[parity].solve(A,b,x=x,parity=parity), parity )
    #===================================
    def save_propagator(self, fname): 
        np.save(fname, self.prop.v_field)
    #===================================
    @staticmethod
    def ks_MdM_operator(U, mass, parity, u0_sq=1):
        """
        .. _MILC code: http://physics.utah.edu/~detar/milc.html

        We follow `Lattice Methods for Quantum Chromodynamics` by Degrand & Detar.
        (See also `mat_invert.c` in `MILC code`_.)

        First, following Eq. (8.66), we define ``M^\dagger M``::
        
                              | m^2 + D D^\dagger         0          |
                M^\dagger M = |                                      |
                              |       0            m^2 + D^\dagger D |

                              | m^2 - D_eo D_oe         0            |
                M^\dagger M = |                                      |
                              |       0            m^2 - D_oe D_eo   |

        where ``D_eo = D = -D_oe^\dagger``.

        Since ``M^\dagger M`` is hermitian positive definite, one could invert the Dirac matrix
        ``M`` by inverting ``M^\dagger M`` and multiplying by ``M^\dagger``.
        This is arguably the optimal choice for staggered fermions (de Forcrand, 1996).
        For the linear system ``Ax = b``, the conjugate gradient algorithm constructs....

        Parameters:
            - ``U``      (numpy.ndarray): the gauge links.
            - ``mass``   (float):   the mass of the quark.
            - ``parity`` (0 or 1):  the parity.
            - ``u0_sq``  (float):   one can change it for tadpole improvement.

        """
        cdef int mu, neta_mu_xx
        cdef int nu, eta_nu_xx
        cdef int ind_xx, num_evensites = num_sites//2
        cdef site XX, YY
        cdef matrix dst_xx
        #cdef matrix M_xx_xx = matrix(identity=True) * 0.25*(4*mass**2 + 2*dim)
        cdef matrix M_xx_xx = matrix(identity=True) * 0.25*(4*mass**2 * u0_sq + 2*dim)
        cdef matrix d_pp, d_nn, d_pn, d_np # p/n: positive/negative
        cdef tuple xx
        row  = []
        col  = []
        data = []
        shape = (num_evensites,num_evensites)
        def append(matrix d, int r, int c):
            if parity==1:
                r -= num_evensites
                c -= num_evensites
            data.append(d/u0_sq)
            row.append(r)
            col.append(c)
        for xx in ALL_sites():
            if sum(xx)%2!=parity:
                continue
            XX = site(*xx)
            ind_xx = XX.index()
            # append( M_xx_xx * u0_sq, ind_xx, ind_xx) # Note that in the append(.) we introduced 1/u0_sq, but it should be complensated here.
            append( M_xx_xx, ind_xx, ind_xx)
            # Important point M_xx_xx should not change under scaling U -> U/u0, even the term that does not depend on the mass
            for mu in range(dim):
                eta_mu_xx = XX.ks_eta(mu)
                for nu in range(mu+1):
                    eta_nu_xx = XX.ks_eta(nu)
                    # A) Four cases as:
                    #       A_pp) +/+ steps in mu/nu directions, respectively
                    #       A_nn) -/-   ...   ...     ...
                    #       A_pn) +/-   ...   ...     ...
                    #       A_np) -/+   ...   ...     ...
                    # B) Note that when nu<mu we also have steps in ``nu/mu`` directions in addition to steps in ``mu/nu`` directions.
                    #    We denote them by ``B_??``, corresponding to ``A_??``, respectively.
                    # C) Using the identity ``eta_mu(XX + YY) = eta_mu(XX) * eta_mu(YY)`` one can show that
                    #       A_??) eta_mu(XX) * eta_nu(XX \pm mu) =  eta_mu(XX) * eta_nu(XX) * eta_nu(mu)
                    #       B_??) eta_nu(XX) * eta_mu(XX \pm nu) =    ...      *  ...       * eta_mu(nu)
                    #    where eta_mu(nu) = -1 if nu<mu otherwise 1.
                    #   
                    # Below we define:
                    #   w_xx = eta_mu_xx * eta_nu_xx * eta_nu_mu
                    #   
                    if nu==mu:
                        YY = XX-mu-mu
                        d_pp =  U[index(XX,mu)] * U[index(XX+mu,mu)]     # A_pp
                        d_nn = (U[index(YY,mu)] * U[index(YY+mu,mu)]).H  # A_nn
                        append( -0.25*d_pp, ind_xx, (XX+mu+nu).index() )
                        append( -0.25*d_nn, ind_xx, YY.index() )
                        # Note that A_pm and A_np are taken care above in M_xx_xx
                    else:# i.e. nu<mu
                        w_xx = eta_mu_xx * eta_nu_xx # note that eta_nu_mu=1 in this case and eta_mu_nu = -1
                        d_pp =  U[index(XX,mu)] * U[index(XX+mu,nu)] - U[index(XX,nu)] * U[index(XX+nu,mu)]     # A_pp + B_pp
                        YY   = XX-mu-nu
                        d_nn = (U[index(YY,nu)] * U[index(YY+nu,mu)] - U[index(YY,mu)] * U[index(YY+mu,nu)]).H  # A_nn + B_nn
                        append( -0.25*w_xx * d_pp, ind_xx, (XX+mu+nu).index() )
                        append( -0.25*w_xx * d_nn, ind_xx, YY.index() )
                        YY   = XX-nu
                        d_pn = U[index(XX,mu)] * U[index(YY+mu,nu)].H - U[index(YY,nu)].H * U[index(YY,mu)]    # A_pn + B_pn
                        YY   = XX-mu
                        d_np = U[index(YY,mu)].H * U[index(YY,nu)]    - U[index(XX,nu)] * U[index(YY+nu,mu)].H # A_np + B_np
                        append( 0.25*w_xx * d_pn, ind_xx, (XX+mu-nu).index() )
                        append( 0.25*w_xx * d_np, ind_xx, (XX-mu+nu).index() )
        MdM_parity = csr_matrix(data, row, col, shape=shape) # (M^dagger * M )_{ee or oo depeneding on parity}
        return MdM_parity
    #===================================

#==========================================================
class quark_base_src(object):
    """ A class to define quark sources.

    One can specify the `srcr_type`, `t0`, `color_list`, `label` and (if meaningful)
    the mommentum of the soruce. 

    """
    def __init__(self, src_type, t0, color_list=[], label="", mom=(0,0,0)):
        self.src_type = src_type
        self.t0       = t0
        self.label    = label
        self.mom      = mom
        self.color_list=color_list
        if len(color_list)>0:
            self.build_src(color_list)   # sets self.v_src
        else:
            self.v_src = []         # can be set later by the ``build_src`` method
    #===================================
    def get_src_kwargs(self, **update_kwargs):
        kwargs = dict(label=self.label, mom=self.mom, color_list=self.color_list)
        kwargs.update(update_kwargs)
        return kwargs
    #===================================
    def build_src(self, color_list):
        self.color_list=color_list
        #cdef numpy.ndarray[object, ndim=1] v_src
        self.v_src = create_vector_field(num_sites)
        for color in color_list:
            if self.src_type=="point_src":      self._point_src(color)
            elif self.src_type=="corner_wall":  self._corner_wall(color)
            elif self.src_type=="even_wall":    self._even_wall(color)
            elif self.src_type=="evenodd_wall": self._evenodd_wall(color)
            elif self.src_type=="evenminusodd_wall": self._evenminusodd_wall(color)
    #===================================
    def _point_src(self, color, x=(0,0,0)):
        t0    = self.t0
        v_src = self.v_src
        for xx in ALL_sites():
            XX = site(*xx)
            if XX.t==t0 and xx[0]==x[0] and xx[1]==x[1] and xx[2]==x[2]:
                v_src[XX.index()][color] = 1.0
    #===================================
    def _corner_wall(self,color):
        t0    = self.t0
        v_src = self.v_src
        for xx in ALL_sites():
            XX = site(*xx)
            if XX.t==t0 and xx[0]%2==0 and xx[1]%2==0 and xx[2]%2==0:
                v_src[XX.index()][color] = 1.0
    #===================================
    def _even_wall(self,color):
        t0    = self.t0
        v_src = self.v_src
        for xx in ALL_sites():
            XX = site(*xx)
            if XX.t==t0:
                if sum(xx)%2 == 0:
                    v_src[XX.index()][color] = 1.0
    #===================================
    def _evenodd_wall(self, color):
        t0    = self.t0
        v_src = self.v_src
        for xx in ALL_sites():
            XX = site(*xx)
            if XX.t==t0:
                v_src[XX.index()][color] = 1.0
    #===================================
    def _evenminusodd_wall(self, color):
        t0    = self.t0
        v_src = self.v_src
        for xx in ALL_sites():
            XX = site(*xx)
            if XX.t==t0:
                if sum(xx)%2 == 0:
                    v_src[XX.index()][color] = 1.0
                else:
                    v_src[XX.index()][color] = -1.0
    #===================================
    def _random_color_wall(self):
        print("NOT implemented yet!")
        #t0 = self.t0
        #v_src = self.v_src
        #for xx in ALL_sites():
        #    XX = site(*xx)
        #    if XX.t==t0:
        #        ind_xx = XX.index()
        #        for color in range(nc):
        #            v_src[ind_xx][color] = rand_complex_normal()
        #        v_src[ind_xx] /= v_src[ind_xx].norm()
    #===================================

#==========================================================
class quark_src(quark_base_src):
    """ This is basically :class:`gauge_tools.util.quark.quark_base_src`
    except that has methods to modify the source.
    """
    #===================================
    def __init__(self, src_type, t0, **kwargs):
        super(quark_src,self).__init__(src_type, t0, **kwargs)
    #===================================
    def get_v_src(self, parity='both'):
        n = len(self.v_src)//2
        if parity=='both':
            return self.v_src
        elif parity==0:
            return self.v_src[:n]
        elif parity==1:
            return self.v_src[n:]
    #===================================
    def evenodd_preconditioner(self, U, mass, u0=1, label=""):
        """ returns a new instance of :class:`gauge_tools.util.quark.quark_src`
        with eve-odd preconditioning."""
        v_src = self.v_src
        if len(v_src)==0:
            raise Exception("Oops: `v_src` is not built yet; first use the build_source method.")
        tmp = self.minus_dslash_mult_src(U, u0=u0)
        new_src = quark_src(self.src_type+"+even/odd_precond", self.t0, **self.get_src_kwargs(label=label))
        for xx in ALL_sites():
            ind_xx = site(*xx).index()
            tmp[ind_xx] += mass * v_src[ind_xx]
        new_src.v_src = tmp
        return new_src 
    #===================================
    def minus_dslash_mult_src(self, U, u0=1):
        #  This method is written following the ``dslash_field`` routine in generic_ks/dslash.c.
        # sets dest. on each site equal to sum of sources parallel transported to site,
        # with minus sign for transport from negative directions.
        v_src = self.v_src
        v_dst = create_vector_field(num_sites)
        for xx in ALL_sites():
            XX = site(*xx)
            v_dst_xx = vector(zeros=True)
            for mu in range(dim):
                v_dst_xx -= U[ index(XX,mu)    ] * v_src[ (XX+mu).index() ]
                v_dst_xx += U[ index(XX-mu,mu) ] * v_src[ (XX-mu).index() ]
            v_dst[ XX.index() ] = v_dst_xx/(2*u0)
        return v_dst
    #===================================

#==========================================================
class propagator(object):
    def __init__(self, src=None):
        self.src = src
        self.v_field = create_vector_field(num_sites)
    #===================================
    def get_v_field(self, parity='both'):
        n = len(self.v_field)//2
        if parity=='both':
            return self.v_field
        elif parity==0:
            return self.v_field[:n]
        elif parity==1:
            return self.v_field[n:]
    #===================================
    def set_v_field(self, v, parity='both'):
        n = len(self.v_field)//2
        i_range = range(n) if parity==0 else range(n,2*n)
        if parity=='both':
            for i in range(n*2):self.v_field[i] = v[i]
        elif parity==0:
            for i in range(n):  self.v_field[i] = v[i]
        elif parity==1:
            for i in range(n):  self.v_field[i+n] = v[i]
    #===================================
    def build_free_prop(self, mass, temporal_bc="periodic", num_images=2):
        """ returns an initial guess for the propagator based on the free field solution of a ``corner_wall`` source. """
        if mass<=0:
            raise Exception("Oops: mass cannot be zero or negative") 
        E = np.arcsinh(mass)
        s = mass
        d = (1+s**2)**0.5
        t0= self.src.t0
        fnc_inf = lambda t: exp(-E*abs(t)) if t>0 or t%2==0 else -exp(-E*abs(t))
        def fnc(t):
            sum_ = 0
            for k in range(-num_images,num_images+1):
                if temporal_bc=="periodic":
                    sum_ += fnc_inf(t+k*nt)
                else:
                    sum_ += fnc_inf(t+k*nt)*(-1)**k
            return sum_/d
        self._fnc = fnc
        if "color_list" in self.src.__dict__.keys():
            color_list = self.src.color_list
        else:
            color_list = [0,1,2]
        g = vector(zeros=True)
        for c in color_list:
            g[c] = 1
        v_field = self.v_field
        for xx in ALL_sites():
            XX = site(*xx)
            if self.src.src_type=="corner_wall" and xx[0]%2==0 and xx[1]%2==0 and xx[2]%2==0:
                v_field[ XX.index() ] = fnc(XX.t - t0) * g
            elif self.src.src_type=="even_wall" and sum(xx)%2==0:
                v_field[ XX.index() ] = fnc(XX.t - t0) * g
            elif self.src.src_type=="evenodd_wall":
                v_field[ XX.index() ] = fnc(XX.t - t0) * g
    #===================================
    @staticmethod
    def ks_project_spatialmom(v_field, mom=(0,0,0), color=None):
        """ Note that 8 of the hyperblock sites at t%2==1 are merged with corresopdong ones at t%2==0. """
        kx,ky,kz = mom
        phase = lambda x,y,z: exp(-(kx*x+ky*y+kz*z)*1J)
        v_field_projected = create_vector_field(nt)
        for x,y,z,t in ALL_sites():
            i = site(x,y,z,t).index()
            v_field_projected[t] += v_field[i]*phase(x,y,z)
        if not isinstance(color,int):
            return v_field_projected
        else:
            return np.array([v[color].real for v in v_field_projected])
    #===================================
    @staticmethod
    def print_vector_field(v_field, n_blocks=None, strides=None, fmt="{:.6f}\t", color=0, realorimag="real"):
        s = ""
        if n_blocks==None or n_blocks>num_sites//16:
            n_blocks = num_sites//16
        if strides==None:
            strides=num_sites//16 # cannot be set as option because num_sites is not defined initially
        if realorimag=="real":
            fnc = lambda x: x.real
        elif realorimag=="imag":
            fnc = lambda x: x.imag
        else:
            fnc = lambda x: x
        for i in range(n_blocks):
            s += "{}:\t".format(i)
            for j in range(8):
                s += fmt.format( fnc(v_field[i+j*strides][color]) )
            s += "\n    \t"
            for j in range(8,16):
                s += fmt.format( fnc(v_field[i+j*strides][color]) )
            s += "\n"
        print(s)
    #===================================
    def avg_data(v_fields_list):
        """ the real/imaginary parts of ``std`` would be the std of the real/imaginary parts of the inputs, respectively. """
        n = len(v_fields_list[0])
        avg = np.mean(v_fields_list,axis=0)
        std = create_vector_field(n)
        for _,v_field in enumerate(v_fields_list):
            for i,v in enumerate(v_field):
                for j in range(3):
                    re = (v[j].real - avg[i][j].real)**2/n
                    im = (v[j].real - avg[i][j].imag)**2/n
                    std[i][j] += (re + 1J*im)
        for i in range(n):
            for j in range(3):
                std[i][j] = std[i][j].real**0.5  +  1J*std[i][j].imag**0.5
        return avg, std

#==========================================================
class cg_solver(object):
    def __init__(self):
        pass
    def solve(self, A, b, x=None, max_iter=300, tol=1e-12, parity='both'):
        T1 = time.time()
        self.max_iter = max_iter
        self.tol = tol
        nr, nc = A.shape
        self.r_norm2_hist = np.zeros(max_iter+1)
        if x is None:
            x = create_vector_field(nc)
        if len(x)!=nc or len(b)!=nr or nr!=nc:
            raise Exception("Oops: the dimensions do not match.")
        vdot    = self.vector_vdot
        norm_sq = self.vector_norm_sq
        r = b - A.dot(x)
        p = r 
        r_norm2_old = norm_sq(r)
        self.r_norm2_hist[0] = r_norm2_old
        header = "CG (parity=%i): "%parity if parity in [0,1] else "CG: "
        if r_norm2_old**0.5<tol:
            print(header + "converged after 0 iterations", end="")
            max_iter = 0
        for k in range(1,1+max_iter):
            p_norm2 = vdot( p, A.dot(p) )
            alpha  = r_norm2_old / p_norm2
            x = x + alpha*p
            r = r - alpha*A.dot(p)
            r_norm2 = norm_sq(r)
            self.r_norm2_hist[k] = r_norm2
            if r_norm2**0.5<tol:
                print(header+"converges after {} iterations; residue={}".format(k,r_norm2**0.5), end="")
                break
            beta = r_norm2 / r_norm2_old
            p = r + beta*p
            r_norm2_old = r_norm2
        else:
            print(header+"does not converge after {} iterations; residue={}".format(k,r_norm2**0.5), end="")
        T2 = time.time()
        print("; time = {}".format(T2-T1))
        return x
    #===================================
    @staticmethod
    def vector_vdot(v1,v2):
        res = 0 
        for i,vec in enumerate(v1):
            res += vec.vdot(v2[i])
        return res
    #===================================
    @staticmethod
    def vector_norm_sq(v):
        res = 0
        for vec in v:
            res += vec.norm_sq()
        return res

#==========================================================
class csr_matrix(object):
    """ complex sparse row matrix; defined after a similar matrix in scipy. """
    def __init__(self, data, row, col, shape=None):
        self.data  = data
        self.row   = row
        self.col   = col
        if shape==None:
            self.nr = np.max(row)
            self.nc = np.max(col)
            shape = (self.nr,self.nc)
        elif isinstance(shape,tuple):
            self.nr    = shape[0]
            self.nc    = shape[1]
        self.shape = shape
    def dot(self, x):
        """ returns ``M*x``. """
        if len(x)!=self.nc:
            raise Exception("Oops: the dimensions do not match.")
        else:
            y = create_vector_field(len(x))
        col  = self.col
        data = self.data
        for i,r in enumerate(self.row):
            c = col[i]
            d = data[i]
            y[r] += d*x[c]
        return y

#==========================================================
def create_vector_field(n=num_sites):
    return np.array([vector(zeros=True) for _ in range(n)])

#==========================================================
