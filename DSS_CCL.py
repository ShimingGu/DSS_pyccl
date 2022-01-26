import numpy as np
import pyccl as ccl
from astropy.convolution import Tophat2DKernel,Ring2DKernel,Gaussian2DKernel
from scipy.optimize import minimize
from scipy.special import jv,gamma
from scipy.integrate import quad

def dU2_func(d0,V):
    return d0**2*V
    
def dU3_func(d0,V):
    return 3*d0*V**2+d0**3*V**3

def sigma_d0_solver(param,dU2,dU3):
    d0,si = param
    if si <= 0 or d0 < 0:
        return np.inf
    else:
        V = np.exp(si**2)-1
        return np.abs(dU2_func(d0,V)-dU2)+np.abs(dU3_func(d0,V)-dU3)

def WRk(Rk): #ToImprove
    return 2*jv(1,Rk)/(Rk)

def NiPoi(N,Nbar):
    return np.exp(N*np.log(Nbar)-Nbar-np.log(gamma(N+1)))

def vecint(dx,fx):
    return np.nansum(dx*0.5*(fx[:,1:]+fx[:,:-1]),axis=1)

def matint(dx,fx):
    return np.nansum(dx*0.5*(fx[:,:,1:]+fx[:,:,:-1]),axis=2)

def quinsep(pdf):
    cdf = np.cumsum(pdf/np.nansum(pdf))
    qui = [0.0,0.2,0.4,0.6,0.8,1.0]
    quinind = np.searchsorted(cdf,qui)
    return quinind

def theta_comoving(Rtchi):
    return Rtchi[:,0]*Rtchi[:,2]

class DSS_tools:
    
    def __init__(self,cosmo,zs,thetas,smoothing_scale,**kwargs):
        self.cosmo = cosmo
        self.zs = zs
        self.Rs = np.deg2rad(smoothing_scale/60.0) #in arcmin
        self.L = 1e10
        self.trick = True
        self.thetas = np.deg2rad(thetas/60.0)
        self._check_kwargs_params(**kwargs)
        self.lk = 1024
        # Compute the S3 (Bernardeau et al. 2002) #Need to be called somewhere in the beginning
        self.S3 = 36.0/7.0 - 1.5*(cosmo['n_s']-1)
        self.reset = True
        self.update = True
        return None
        
# Checking keyword parameter arguments
    def _check_kwargs_params(self,**kwargs):
        thetas = self.thetas
        self.pst = kwargs.pop('p_st',0.3)
        self.qst = kwargs.pop('q_st',0.707)
        self.bmass = kwargs.pop('b_mass',0.8)
        self.P0 = kwargs.pop('P_0',6.41)
        self.bgal = kwargs.pop('b_gal',1.618)
        self.rgal = kwargs.pop('r_gal',0.956)
        self.Ngal = kwargs.pop('N_gal',2.5e-2)#9.0*np.pi*(60*np.rad2deg(self.Rs))**2)
        # Create Angular Bins
        if len(thetas) == 3:
            tmin = thetas[0];tmax = thetas[1];tnum = thetas[2]
            if kwargs['theta_space']:
                if kwargs['theta_space'] == 'linear':
                    angbins = np.linspace(tmin,tmax,3*tnum)
                elif kwargs['theta_space'] == 'log':
                    angbins = np.geomspace(tmin,tmax,3*tnum)
            else:
                angbins = np.geomspace(tmin,tmax,3*tnum)
            if kwargs['theta_notrick']:
                self.angmins = angbins[0::3]
                self.angcent = angbins[1::3]
                self.angmaxs = angbins[2::3]
                self.trick = False
            else:
                angcent = angbins[1::3]
        else:
            angcent = thetas
        if self.trick == True:
            self.angmins = angcent*0.995
            self.angcent = angcent
            self.angmaxs = angcent*1.005
        
        self.smoothing_kernel = kwargs.pop('kernel','Tophat')# Kernel Keyword Arguments
        self.bg = kwargs.pop('bg','Gamma_t')# Background Keyword Arguments
        self.fg = kwargs.pop('fg','N_ap')# Foreground Keyword Arguments

    # Setup Halo Models, Some parameters can be given including p, q, mass bias, pressure normalisation factor
    def halo_setups(self):#,p=0.3,q=0.707,bmass=0.8,P0=6.41):
        #self.pst = p; self.qst = q
        #self.bmass = bmass; self.P0 = P0
        self.karr = np.geomspace(1e-4,1e2,self.lk)
        self.dkar = np.diff(self.karr)
        #self.aarr = np.linspace(0.2,1,64)
        self.aarr = 1/(1.0+self.zs)
        #self.aarr = np.linspace(0.5,1,2)

        self.hmdefi = ccl.halos.MassDefVir()
        self.hmconc = ccl.halos.ConcentrationDuffy08(self.hmdefi)
        self.hmfunc = ccl.halos.hmfunc.MassFuncSheth99(self.cosmo,self.hmdefi,p=self.pst,q=self.qst)
        self.hmbias = ccl.halos.hbias.HaloBiasSheth99(self.cosmo,self.hmdefi,p=self.pst,q=self.qst)
        self.hmcalc = ccl.halos.halo_model.HMCalculator(self.cosmo,self.hmfunc,self.hmbias,self.hmdefi,log10M_min=6.0,log10M_max=17.0)
        return None

    # Calculate the Mass and Pressure Profiles
    def profiles(self):
        self.hmprof = ccl.halos.profiles.HaloProfileNFW(self.hmconc)
        self.gpprof = ccl.halos.profiles.HaloProfilePressureGNFW(mass_bias=self.bmass,P0=self.P0)
        return None

    # Calculate Auto/Cross-Spectra, result self.Pkmm3D and self.Pkmp3D if needed
    def Pks(self):
        try:
            self.profiles()
        except:
            print ('Halo Model Calculations Not Setup')
            raise ('ValueError')
        else:
            print ('Halo Model Calculations Set')
        #if self.Pk_type == 'MP' or 'PM':
        self.Pkmp3D = ccl.halos.halo_model.halomod_power_spectrum(cosmo=self.cosmo,hmc=self.hmcalc,k=self.karr,a=self.aarr,
                                                   prof=self.hmprof,prof2=self.gpprof,normprof1=True,normprof2=True)
        self.Pkmm3D = ccl.halos.halo_model.halomod_power_spectrum(cosmo=self.cosmo,hmc=self.hmcalc,k=self.karr,a=self.aarr,
                                                   prof=self.hmprof,normprof1=True)
        # double (lk)
        return None

    # Calculate the integrand for variance computation
    def var_vec(self,ks,Ra):
        Rz = Ra*self.chis
        ZK = np.array(np.meshgrid(Rz,ks)).T.reshape(-1,2)
        return (ZK[:,1]*WRk(ZK[:,0]*ZK[:,1])**2).reshape(self.zl,self.lk) # double (zl,lk)

    # Compute the variance of the cylinder of given chi,R and L -> 1dim array as func of chi
    def variance_mm(self):
        vi = self.var_vec(self.karr,self.Rs)*self.Pkmm3D
        ii = vecint(self.dkar,vi)
        self.var = ii*(2*np.pi)**2#*(self.Dp2chi)**2
        return None

    # Calculate the integrand for the covariance
    def cvr_mat(self,ks,R1s,R2): # to finish tomorrow
        R1kchi = np.array(np.meshgrid(R1s,ks,self.chis)).T.reshape(-1,3) 
        # must be organised as meshgrid(x,z,y) otherwise numpy.array.reshape don't give correct answer
        Rz1 = theta_comoving(R1kchi) #.reshape(self.Rtl,self.zl,self.lk) # double (Rtl*zl*lk)
        Rz2 = R2*R1kchi[:,2] # double (Rtl*zl*lk)
        return (R1kchi[:,1]*WRk(Rz1*R1kchi[:,1])*WRk(Rz2*R1kchi[:,1])).reshape(self.Rtl,self.zl,self.lk)

    # Compute the matter-matter covariance of the cylinder of given chi,R and L -> 1dim array as func of chi into 2d array
    def covariance_mm(self):
        vi = self.cvr_mat(self.karr,self.Rt,self.Rs)*self.Pkmm3D
        ii = matint(self.dkar,vi)
        self.cvr = ii*(2*np.pi)**2
        #Di = np.array(np.meshgrid(self.Dp2chi,ii)).T.reshape(-1,2)
        #self.cvr = (Di[:,1]*(2*np.pi)**2*(Di[:,0])**2).reshape(self.Rtl,self.zl)
        return None
        # 1 dim array as a func of chi

    # Compute the matter-pressure covariance of the cylinder of given chi,R and L -> 1dim array as func of chi into 2d array
    def covariance_mp(self):
        vi = self.cvr_mat(self.karr,self.Rt,self.Rs)*self.Pkmp3D
        ii = matint(self.dkar,vi)
        self.cvr = ii*(2*np.pi)**2
        #Di = np.array(np.meshgrid(self.Dp2chi,ii)).T.reshape(-1,2)
        #self.cvr = (Di[:,1]*(2*np.pi)**2*(Di[:,0])**2).reshape(self.Rtl,self.zl)
        return None

    # Compute the skewness by S3 -> 1 dim array as func of chi
    def skewness_mm(self):
        self.skw = self.S3*self.var**2
        return None

    # Compute the Coskewness by S3 -> 1 dim array as func of chi into 2d array (not precise!!!) #ToImprove
    def coskewness_mm(self):
        self.csw = self.S3*self.cvr*self.var
        return None

    # Calculate the projection integrands for the variance
    def moment_integrand(self,dchi,tracer,moment):
        vi = tracer*moment
        ii = np.nansum(dchi*0.5*vi[:-1]+vi[1:])
        return ii

    def moment_integrand_vec(self,dchi,tracer,moment):
        vi = tracer*moment
        ii = np.nansum(dchi*0.5*vi[:,:-1]+vi[:,1:],axis=1)
        return ii

    # Integrate to get moments interpretable
    def momentchi(self):
        self.zl = len(self.zs);rtl = self.Rtl
        az = 1./(1.+self.zs)
        self.chis = ccl.background.comoving_radial_distance(self.cosmo,az)
        dchi = np.diff(self.chis)
        #self.Dp2chi = ccl.background.growth_factor(self.cosmo,az) # not used now, will use P(k,z) instead

        if self.reset == True:
            #self.var = np.zeros(zl)
            self.variance_mm()
            self.skewness_mm()
            # Galaxy Kernel # Need further comparisons
            chi,self.fgw = ccl.tracers.get_density_kernel(self.cosmo,(self.zs,self.nz))
            self.deltaU2_var = self.moment_integrand(dchi,self.fgw**2,self.var)
            self.deltaU3_skew= self.moment_integrand(dchi,self.fgw**3,self.skw)
            # self.reset = False
            
        if self.update == True:
            #self.cvr = np.zeros((rtl,zl))
            if self.bg == 'Gamma_t': # Lensing Kernel
                chi,self.bgw = ccl.tracers.get_lensing_kernel(self.cosmo,(self.zs,self.nz))
                self.covariance_mm()
            elif self.bg == 'y': # tSZ Kernel
                self.bgw = 4.01710079e-06*az
                self.covariance_mp()
            elif self.bg == 'cmbl':
                chi,self.bgw = ccl.tracers.get_kappa_kernel(self.cosmo,(self.zs,self.nz),500)
                self.covariance_mm()
            #self.update = False
            self.coskewness_mm()
            self.I_deltaU_cov = self.moment_integrand_vec(dchi,self.bgw*self.fgw,self.cvr)
            self.I_deltaU2_cosk = self.moment_integrand_vec(dchi,self.bgw*self.fgw**2,self.csw)

        return None
        
    # Get sigma and deltaU_0 for pdfs -> self.pdf_sigma & self.deltaU_0
    def sigma_d0_obtainer(self):
        dU2 = self.deltaU2_var
        dU3 = self.deltaU3_skew
        d0,si = minimize(sigma_d0_solver,(0.05,0.05),args=(dU2,dU3),tol=1e-5).x
        self.pdf_sigma = si
        self.deltaU_0 = d0
        return None

    # Compute Foreground PDF -> double self.pdU_arg
    def pdU_compute(self,dU):
        sig = self.pdf_sigma
        dt0 = self.deltaU_0
        #self.pdU_arg = (1/(np.sqrt(2*np.pi)*sig*(dU+dt0)))*np.exp(-((np.log(dU/dt0+1)+0.5*sig**2)**2)/(2*sig**2))
        return (1/(np.sqrt(2*np.pi)*sig*(dU+dt0)))*np.exp(-((np.log(dU/dt0+1)+0.5*sig**2)**2)/(2*sig**2))

    # Compute Joint Foreground-Background PDF -> double(Rtl,ldU) self.IdU_arg_Rt
    def IdU_compute(self,dU):
        ldU = len(dU)
        Rtl = self.Rtl
        sig = self.pdf_sigma
        dt0 = self.deltaU_0
        IdU = self.I_deltaU_cov     #double (Rtl)
        IdU2 = self.I_deltaU2_cosk  #double (Rtl)
        I0dU = (IdU*IdU*np.exp(sig**2))/(IdU2-2*IdU*dt0*(np.exp(sig**2)-1)) #double (Rtl)
        C0 = np.log(1+(IdU/(dt0*I0dU))) #double (Rtl)
        C0dU = np.array(np.meshgrid(C0,dU)).T.reshape(-1,2)
        meshpart = (C0dU[:,0]*(2*np.log(C0dU[:,1]/dt0+1)+sig**2-C0dU[:,0])).reshape(Rtl,ldU)
        return (I0dU*np.exp((meshpart.T/(2*sig**2))-1)).T #double (Rtl,ldU)
        #self.IdU_arg_Rt = I0dU*np.exp((C0*(2*np.log(dU/dt0+1)+sig**2-C0)/(2*sig**2))-1) #double (Rtl,ldU) # not working
        #return I0dU*np.exp((C0*(2*np.log(dU/dt0+1)+sig**2-C0)/(2*sig**2))-1) #double (Rtl,ldU) # Redo this line

    # Compute the Conditional PDF of the Foreground Galaxy Aperature Number Counts -> double
    def P_Nap_dU(self,NdU):
        return NiPoi(NdU[:,0],self.Ngal*(1+self.bgal*NdU[:,1]))
    
    # Compute the Conditional PDF of the Foreground Convergence Aperature Masses #todolist
    def P_Map_dU(self,MdU):
        return 1

    # Compute the Conditional PDF of the Foreground -> double self.pFdU_arg
    def pFdU_select(self,F,dU):
        FdU = np.array(np.meshgrid(F,dU)).T.reshape(-1,2)
        if self.fg == 'N_ap':
            return self.P_Nap_dU(FdU)
        elif self.fg == 'M_ap':
            return self.P_Map_dU(FdU)

    # Split the Quintiles
    def quintilecomp(self,q): # I don't want to use for loops but 5 should be fast enough
        iFmin = self.quintile_indices[q-1]
        iFmax = self.quintile_indices[q]
        ddF = np.diff(self.dFs[iFmin:iFmax])
        ifenmu = self.pFs[iFmin:iFmax]
        ifenzi = ifenmu*self.IFs[:,iFmin:iFmax]
        fenzi = vecint(ddF,ifenzi)
        fenmu = 1.0*np.nansum(0.5*ddF*(ifenmu[1:]*ifenmu[:-1]))
        return fenzi/fenmu

    # Compute Mean-Inside-Profiles, in terms of double(Rtl,5) #tofinish
    def posterior_cal(self,thetas):
        ldUs = 2000;self.Rt = thetas;self.Rtl = len(self.Rt)
        self.momentchi()
        self.sigma_d0_obtainer()
        dUs = np.linspace(0,5,ldUs)-self.deltaU_0+1e-3
        self.dUs = dUs
        ddU = np.diff(dUs)
        if self.reset == True:
            try:
                self.Pkmm3D
            except:
                self.Pks()
            self.dFs = self.Ngal*(self.bgal*dUs+1) # toupdate with if clause, current one only applies for galaxies
            ldFs = len(self.dFs)
            self.pFdUsnn = self.pFdU_select(self.dFs,dUs).reshape(ldFs,ldUs) # be careful, need [i::ldUs] to pick every F with same dU and [i*ldUs:(i+1)*ldUs] to pick every dU
            self.pFdUs = self.pFdUsnn/np.nansum(self.pFdUsnn)
            self.pdUsnn = self.pdU_compute(dUs)
            self.pdUs = self.pdUsnn/np.nansum(self.pdUsnn)
            ipFs = self.pFdUs*self.pdUs
            self.pFsnn = vecint(ddU,ipFs) # double (ldFs)
            self.pFs = self.pFsnn/np.nansum(self.pFsnn)
            self.quintile_indices = quinsep(self.pFs) # quintile index associated with pFs, keep it in index which speeds up further calculations
            self.pdUFs = (self.pFdUs*self.pdUs).T/self.pFs # double (ldUs,ldFs)
            self.reset = False
        if self.update == True:
            self.IdUs = self.IdU_compute(dUs).reshape(self.Rtl,ldUs)
        self.IFs = np.dot(self.IdUs,self.pdUFs) # sum over all possible dUs by matrix multiplication
        return np.array([self.quintilecomp(1),self.quintilecomp(2),self.quintilecomp(3),self.quintilecomp(4),self.quintilecomp(5)])    

    # Organize the posterior computation, compute with angmaxs, angcent and angmins
    def posterior_org(self):
        self.MIP_max = self.posterior_cal(self.angmaxs)
        self.MIP_cen = self.posterior_cal(self.angcent)
        self.MIP_min = self.posterior_cal(self.angmins)
        self.update = False
        return None

    # Compute observables
    def stat(self,zs,nzs,reset=False,update=False,bg=None):
        self.reset = reset;self.update = update
        if update == True:
            if bg != None:
                print ('Update Background Tracer')
                self.bg = bg
        try:
            I_cent = self.MIP_cen
        except:
            self.zs = zs
            self.nz = nzs
            self.posterior_org()

        I_maxs = self.MIP_max;I_mins = self.MIP_min;I_cent = self.MIP_cen       # Mean Inside Profile
        T_maxs = self.angmaxs;T_mins = self.angmins;T_cent = self.angcent       # Angular Bins 
        I_diff = (T_cent/2)*(I_maxs-I_mins)/(T_maxs-T_mins)
        if self.bg == 'Gamma_t':
            B_vath = -I_diff
        else:
            B_vath = I_cent+I_diff

        return np.rad2deg(T_cent)*60.0,B_vath


            
        
        
        