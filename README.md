# DSS_pyccl
A fast code that borrows pyccl to compute density split statistics with different foregrounds and backgrounds


cosmo = ccl.Cosmology(...)

dss = DSS_tools(cosmo,z,theta,smoothing_scale,**kwargs)

dss.halo_setups()

dss.stat(z,nz,reset=True,update=True) 
# gives angular scales and corresponding density split statistics
