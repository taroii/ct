# uses He-Yuan predictor corrector
# rho = 1 is the normal CP

from pylab import *
ion()

from numpy import *
import numpy as np
eig = linalg.eig
eigvals = linalg.eigvals
svd = linalg.svd
inv = linalg.inv
pinv = linalg.pinv
import time

import numba
from numba import njit

import scipy
from scipy import ndimage
gfilt =ndimage.gaussian_filter

resultsfile = "dataHan256/"
mfact = 2       #specifies size of the phantom image. The phantom image size will be (512/mfact)^2
cutoffparm = 4.0  # parameter for the sqrt-hanning filter

#load in a test image
imagenumber = 3  # selection range is 0 through 9
material = "Adipose"  # options are "Adipose", "Fibroglandular" and "Calcification"
phantom1 =load("Phantom_"+material+".npy")[imagenumber]
material = "Fibroglandular" 
phantom2 =load("Phantom_"+material+".npy")[imagenumber]
material = "Calcification" 
phantom3 =load("Phantom_"+material+".npy")[imagenumber]

testimage = (0.5*phantom1 + 1.0*phantom2 + 2.0*phantom3).astype("float64")   #Build our testimage
testimage = testimage[::mfact,::mfact]*1.

# searched optimal parameter settings for different phantom sizes
# 128, alpha=1.95 , beta = 10
# 256, alpha=1.9  , beta = 10
# 512, alpha=1.7  , beta = 5

addnoise = 0
nph = 1.e6
nuxfact = 0.5 #combination factor for x-gradient with projection matrix
nuyfact = 0.5 #combination factor for y-gradient with projection matrix
l1f = 1.0 #don't change this parameter yet
eps = 0.001 # data error constraint in terms of RMSE
larc = 1.0 #1.0 (if limited angular range)  or 0.0 (full 360 scan)
alpha = 1.75  #combination strength of x- and y- gradients
beta = 5.0
rho = 1.75 #over-relaxation parameter for He-Yuan predictor corrector

stepbalance = 100.0 # step-size ratio for CP, needs to be tuned 

#reduce phantom size
phimage = testimage*1.
testimage = phimage*0.
bpimage = testimage*0.

#image parameters
ximageside = 10.0   #cm
yimageside = 10.0   #cm
nx = int(512/mfact)
ny = int(512/mfact)
npix = nx*ny
dx = ximageside/nx
dy = yimageside/ny
xar=arange(-ximageside/2. + dx/2 , ximageside/2., dx)[:,newaxis]*ones([ny])
yar=ones([nx,ny])*arange(-yimageside/2. + dy/2 , yimageside/2., dy)
rar=sqrt(xar**2 + yar**2)
mask = phimage*0.
mask[rar<=ximageside/2.]=1.
#mask.fill(1.)

# sinogram parameters. Note that there are multiple names for some
# of the parameters. This is due to legacy code.
radius = 50.0    #cm
source_to_detector = 100.0   #cm
srad = radius
sd = source_to_detector
slen = (50./180.)*pi     #angular range of the scan
slen0 = -slen/2.0
ns0 = 25
nu0 = 1024
nviews = ns0   
nbins = nu0
nrays = nbins*nviews
epssc = eps*sqrt(nrays)


# The linear detector length is computed in the projection function so that it is
# the exact size needed to capture the projection of the largest inscribed circle in
# the image array.
fanangle2 = arcsin((ximageside/2.)/radius)  # This only works for ximageside = yimageside
detectorlength = 2.*tan(fanangle2)*source_to_detector
u0 = -detectorlength/2.

du = detectorlength/nbins
ds = slen/(nviews-larc)
dup = du*radius/source_to_detector  #detector bin spacing at iso-center



sinogram=zeros([nviews,nbins])

#@njit(cache=True) # If computing projection multiple times use: @njit(cache=True)
@njit
###############
def circularFanbeamProjection(image,sinogram, 
                              nx = nx, ny = ny, ximageside = ximageside, yimageside = yimageside,
                              radius = srad, source_to_detector = sd, detectorlength = detectorlength,
                              nviews = ns0, slen = slen, slen0 = slen0, nbins = nu0):


   dx = ximageside/nx
   dy = yimageside/ny
   x0 = -ximageside/2.
   y0 = -yimageside/2.

   #compute length of detector so that it views the inscribed FOV of the image array
   u0 = -detectorlength/2.

   du = detectorlength/nbins
   ds = slen/(nviews-larc)

   for sindex in range(nviews):
#      print("Doing view number: ",sindex)   #UNCOMMENT if you want to see view progress
      s = sindex*ds+slen0
# Location of the source
      xsource=radius*cos(s)
      ysource=radius*sin(s)

# detector center
      xDetCenter=(radius - source_to_detector)*cos(s)
      yDetCenter=(radius - source_to_detector)*sin(s)

# unit vector in the direction of the detector line
      eux = -sin(s)
      euy =  cos(s)

# Unit vector in the direction perpendicular to the detector line
      ewx = cos(s)
      ewy = sin(s)

      for uindex in range(nbins):

         u = u0+(uindex+0.5)*du
         xbin = xDetCenter + eux*u
         ybin = yDetCenter + euy*u

         xl=x0
         yl=y0

         xdiff=xbin-xsource
         ydiff=ybin-ysource
         xad=abs(xdiff)*dy
         yad=abs(ydiff)*dx

         if (xad>yad):   # loop through x-layers of image if xad>yad. This ensures ray hits only one or two pixels per layer
            slope=ydiff/xdiff
            travPixlen=dx*sqrt(1.0+slope*slope)
            yIntOld=ysource+slope*(xl-xsource)
            iyOld=int(floor((yIntOld-y0)/dy))
            raysum=0.
            for ix in range(nx):
               x=xl+dx*(ix + 1.0)
               yIntercept=ysource+slope*(x-xsource)
               iy=int(floor((yIntercept-y0)/dy))
               if iy == iyOld: # if true, ray stays in the same pixel for this x-layer
                  if ((iy >= 0) and (iy < ny)):
                     raysum=raysum+travPixlen*image[ix,iy]
               else:    # else case is if ray hits two pixels for this x-layer
                  yMid=dy*max(iy,iyOld)+yl
                  ydist1=abs(yMid-yIntOld)
                  ydist2=abs(yIntercept-yMid)
                  frac1=ydist1/(ydist1+ydist2)
                  frac2=1.0-frac1
                  if ((iyOld >= 0) and (iyOld < ny)):
                     raysum = raysum+frac1*travPixlen*image[ix,iyOld]
                  if ((iy>=0) and (iy<ny)):
                     raysum=raysum+frac2*travPixlen*image[ix,iy]
               iyOld=iy
               yIntOld=yIntercept
         else: # loop through y-layers of image if xad<=yad
            slopeinv=xdiff/ydiff
            travPixlen=dy*sqrt(1.0+slopeinv*slopeinv)
            xIntOld=xsource+slopeinv*(yl-ysource)
            ixOld=int(floor((xIntOld-x0)/dx))
            raysum=0.
            for iy in range(ny):
               y=yl+dy*(iy + 1.0)
               xIntercept=xsource+slopeinv*(y-ysource)
               ix=int(floor((xIntercept-x0)/dx))
               if (ix == ixOld): # if true, ray stays in the same pixel for this y-layer
                  if ((ix >= 0) and (ix < nx)):
                     raysum=raysum+travPixlen*image[ix,iy]
               else:  # else case is if ray hits two pixels for this y-layer
                  xMid=dx*max(ix,ixOld)+xl
                  xdist1=abs(xMid-xIntOld)
                  xdist2=abs(xIntercept-xMid)
                  frac1=xdist1/(xdist1+xdist2)
                  frac2=1.0-frac1
                  if ((ixOld >= 0) and (ixOld < nx)) :
                     raysum=raysum+frac1*travPixlen*image[ixOld,iy]
                  if ((ix>=0) and (ix<nx)) :
                     raysum=raysum+frac2*travPixlen*image[ix,iy]
               ixOld=ix
               xIntOld=xIntercept
         sinogram[sindex,uindex]=raysum



@njit(cache=True) # If computing projection multiple times use: @njit(cache=True)
def circularFanbeamBackProjection(sinogram, image,
                              nx = nx, ny = ny, ximageside = ximageside, yimageside = yimageside,
                              radius = srad, source_to_detector = sd, detectorlength  = detectorlength,
                              nviews = ns0, slen = slen,slen0=slen0, nbins = nu0):

   image.fill(0.)

   dx = ximageside/nx
   dy = yimageside/ny
   x0 = -ximageside/2.
   y0 = -yimageside/2.

   u0 = -detectorlength/2.

   du = detectorlength/nbins
   ds = slen/(nviews - larc)

   for sindex in range(nviews):
#      print("Doing view number: ",sindex)   #UNCOMMENT if you want to see view progress
      s = sindex*ds + slen0
# Location of the source
      xsource=radius*cos(s)
      ysource=radius*sin(s)

# detector center
      xDetCenter=(radius - source_to_detector)*cos(s)
      yDetCenter=(radius - source_to_detector)*sin(s)

# unit vector in the direction of the detector line
      eux = -sin(s)
      euy =  cos(s)

# Unit vector in the direction perpendicular to the detector line
      ewx = cos(s)
      ewy = sin(s)

      for uindex in range(nbins):

         sinoval = sinogram[sindex,uindex]
         u = u0+(uindex+0.5)*du
         xbin = xDetCenter + eux*u
         ybin = yDetCenter + euy*u

         xl=x0
         yl=y0

         xdiff=xbin-xsource
         ydiff=ybin-ysource
         xad=abs(xdiff)*dy
         yad=abs(ydiff)*dx

         if (xad>yad):   # loop through x-layers of image if xad>yad. This ensures ray hits only one or two pixels per layer
            slope=ydiff/xdiff
            travPixlen=dx*sqrt(1.0+slope*slope)
            yIntOld=ysource + slope*(xl-xsource)
            iyOld=int(floor((yIntOld-y0)/dy))
            for ix in range(nx):
               x=xl+dx*(ix + 1.0)
               yIntercept=ysource+slope*(x-xsource)
               iy=int(floor((yIntercept-y0)/dy))
               if iy == iyOld: # if true, ray stays in the same pixel for this x-layer
                  if ((iy >= 0) and (iy < ny)):
                     image[ix,iy] =image[ix,iy]+sinoval*travPixlen
               else:    # else case is if ray hits two pixels for this x-layer
                  yMid=dy*max(iy,iyOld)+yl
                  ydist1=abs(yMid-yIntOld)
                  ydist2=abs(yIntercept-yMid)
                  frac1=ydist1/(ydist1+ydist2)
                  frac2=1.0-frac1
                  if ((iyOld >= 0) and (iyOld < ny)):
                     image[ix,iyOld] =image[ix,iyOld]+frac1*sinoval*travPixlen
                  if ((iy>=0) and (iy<ny)):
                     image[ix,iy] =image[ix,iy]+frac2*sinoval*travPixlen
               iyOld=iy
               yIntOld=yIntercept
         else: # loop through y-layers of image if xad<=yad
            slopeinv=xdiff/ydiff
            travPixlen=dy*sqrt(1.0+slopeinv*slopeinv)
            xIntOld=xsource+slopeinv*(yl-ysource)
            ixOld=int(floor((xIntOld-x0)/dx))
            for iy in range(ny):
               y=yl+dy*(iy + 1.0)
               xIntercept=xsource+slopeinv*(y-ysource)
               ix=int(floor((xIntercept-x0)/dx))
               if (ix == ixOld): # if true, ray stays in the same pixel for this y-layer
                  if ((ix >= 0) and (ix < nx)):
                     image[ix,iy] =image[ix,iy]+sinoval*travPixlen
               else:  # else case is if ray hits two pixels for this y-layer
                  xMid=dx*max(ix,ixOld)+xl
                  xdist1=abs(xMid-xIntOld)
                  xdist2=abs(xIntercept-xMid)
                  frac1=xdist1/(xdist1+xdist2)
                  frac2=1.0-frac1
                  if ((ixOld >= 0) and (ixOld < nx)) :
                     image[ixOld,iy] =image[ixOld,iy]+frac1*sinoval*travPixlen
                  if ((ix>=0) and (ix<nx)) :
                     image[ix,iy] =image[ix,iy]+frac2*sinoval*travPixlen
               ixOld=ix
               xIntOld=xIntercept

@njit
###############
def circularFanbeamPDBackProjection(sinogram,image, fov_radius = ximageside/1.,
       nx = nx, ny = ny, ximageside = ximageside, yimageside = yimageside,
       radius = radius, source_to_detector = source_to_detector,
       nviews = nviews, slen = slen, slen0 = slen0, nbins = nbins, detectorlength = detectorlength):
# This implementation is for weighted pixel-driven back-projection needed for fan-beam FBP

   dx = ximageside/nx
   dy = yimageside/ny
   x0 = -ximageside/2.
   y0 = -yimageside/2.

   u0 = -detectorlength/2.

   du = detectorlength/nbins
   ds = slen/(nviews -larc)

   for sindex in range(nviews):
#      print("Doing view number: ",sindex)   #UNCOMMENT if you want to see view progress
      s = sindex*ds + slen0
# Location of the source
      xsource=radius*cos(s)
      ysource=radius*sin(s)

# detector center
      xDetCenter=(radius - source_to_detector)*cos(s)
      yDetCenter=(radius - source_to_detector)*sin(s)

# unit vector in the direction of the detector line
      eux = -sin(s)
      euy =  cos(s)

# Unit vector in the direction perpendicular to the detector line
      ewx = cos(s)
      ewy = sin(s)


      for iy in range(ny):
         pix_y = y0 + dy*(iy+0.5)
         for ix in range(nx):
            pix_x = x0 + dx*(ix+0.5)

            frad = sqrt(pix_x**2. + pix_y**2.)
            fphi = arctan2(pix_y,pix_x)
            if (frad<=fov_radius):

               ew_dot_source_pix = (pix_x-xsource)*ewx + (pix_y-ysource)*ewy        
               rayratio = -source_to_detector/ew_dot_source_pix

               det_int_x = xsource+rayratio*(pix_x-xsource)
               det_int_y = ysource+rayratio*(pix_y-ysource)

               upos = ((det_int_x-xDetCenter)*eux +(det_int_y-yDetCenter)*euy)
               if (upos-u0 >= du/2.) and (upos-u0 < detectorlength-du/2.):
                  bin_loc = (upos-u0)/du +0.5
                  nbin1 = int(bin_loc) -1
                  nbin2 = nbin1+1 
                  frac= bin_loc - int(bin_loc)
                  det_value=frac*sinogram[sindex,nbin2]+(1.-frac)*sinogram[sindex,nbin1]
               else:
                  det_value = 0.0

               image[ix,iy] += det_value*ds

def euclidean_proj_simplex(v, s=1):
   """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
   assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
   n, = v.shape  # will raise ValueError if v is not 1-D
   # check if we are already on the simplex
   if v.sum() == s and np.alltrue(v >= 0):
      # best projection: itself!
      return v
   # get the array of cumulative sums of a sorted (decreasing) copy of v
   u = np.sort(v)[::-1]
   cssv = np.cumsum(u)
   # get the number of > 0 components of the optimal solution
   rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
   # compute the Lagrange multiplier associated to the simplex constraint
   theta = (cssv[rho] - s) / (rho + 1.0)
   # compute the projection by thresholding v using theta
   w = (v - theta).clip(min=0)
   return w


def euclidean_proj_l1ball(v, s=1):
   """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
   Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
   assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
   n, = v.shape  # will raise ValueError if v is not 1-D
   # compute the vector of absolute values
   u = np.abs(v)
   # check if v is already a solution
   if u.sum() <= s:
       # L1-norm is <= s
       return v
   # v is not already a solution: optimum lies on the boundary (norm == s)
   # project *u* on the simplex
   w = euclidean_proj_simplex(u, s=s)
   # compute the solution to the original problem on v
   w *= np.sign(v)
   return w

#generate gradient matrix for directional TVs
gmatx = zeros([nx,nx])
for i in range(nx):
   gmatx[i,i] = -1.0
for i in range(nx-1):
   gmatx[i,i+1] = 1.0

gmaty = zeros([ny,ny])
for i in range(ny):
   gmaty[i,i] = -1.0
for i in range(ny-1):
   gmaty[i,i+1] = 1.0
def gradx(image):
   return dot(gmatx,image)
def grady(image):
   return array(dot(gmaty,image.T).T,order="C")
def mdivx(image):
   return dot(gmatx.T,image)
def mdivy(image):
   return array(dot(gmaty.T,image.T).T,order="C")

#gradient implementations for TV
def gradim(image):

   xgrad = image.copy()
   ygrad = image.copy()
   temp = image
   xgrad[:-1,:] = temp[1:,:] - temp[:-1,:]
   ygrad[:,:-1] = temp[:,1:] - temp[:,:-1]
   xgrad[-1,:] =  -1.0* temp[-1,:]
   ygrad[:,-1] =  -1.0* temp[:,-1]

   return xgrad,ygrad

def mdiv(xgrad,ygrad):
   divim = xgrad.copy()
   shp = [xgrad.shape[0] + 2, xgrad.shape[1] +2]
   xgradp=zeros(shp)
   ygradp=zeros(shp)
   xgradp[1:-1,1:-1] = xgrad*1.
   ygradp[1:-1,1:-1] = ygrad*1.
   divim.fill(0.)
   divim = xgradp[:-2,1:-1] + ygradp[1:-1,:-2] - xgradp[1:-1,1:-1] - ygradp[1:-1,1:-1]

   return divim



if 1==1:
   nb0 = nbins
   blen0 =  detectorlength
   db = blen0/nb0
   b00 = - blen0/2.
   uar = arange(b00 + db/2., b00 + blen0, db)*1.
   uhanp = abs(b00)/cutoffparm
   uhan = 0.5*(1.0 + cos(pi*uar/uhanp))
   uhan[abs(uar)>uhanp] = 0.
#   filterfun = abs(uar)*uhan
   filterfun = abs(uar)

def fo(sinom):
   imft = fft.fft(sinom,axis=1)
   pimft = (ones([nbins])*fft.fftshift(sqrt(filterfun)))*imft
   res = (1.*fft.ifft(pimft,axis=1).real)
   return res

#########BEGIN DATA GENERATION #############

truesino=sinogram*0.
circularFanbeamProjection(phimage,truesino)
#add noise
if addnoise:
   sinodata =-log(poisson(nph*exp(-truesino))/nph)
else:
   sinodata= truesino*1.

sinodata = fo(sinodata)

xgrad = gradx(phimage)
gim = sqrt(xgrad**2)
truetvx = gim.sum()

ygrad = grady(phimage)
gim = sqrt(ygrad**2)
truetvy = gim.sum()


xgrad,ygrad = gradim(phimage)
gim = sqrt(xgrad**2 + ygrad**2)
truetv = gim.sum() #calc true TV


# compute norm of X
xim = randn(nx,ny)
xim *= mask
npower = 50
worksino =truesino*0.
for i in range(npower):
   circularFanbeamProjection(xim,worksino)
   worksino = fo(worksino)
   xim.fill(0.)
   worksino = fo(worksino)
   circularFanbeamBackProjection(worksino,xim)
   xim *= mask
   xnorm2 = sqrt( (xim**2.).sum() )
   xim/=xnorm2
snorm = sqrt(xnorm2)
nusino = 1./snorm
print("nusino: ",nusino)
#xvec = xim*1.
input("hi")

# compute norm of Dx
xim = randn(nx,ny)
xim *= mask
npower = 50
for i in range(npower):
   xg = gradx(xim)
   xim.fill(0.)
   xim  = mdivx(xg)
   xim *= mask
   xnorm2 = sqrt( (xim**2.).sum() )
   xim/=xnorm2
gnorm = sqrt(xnorm2)
nuxgrad = nuxfact/gnorm
print("nuxgrad: ",nuxgrad)
#dxvec = xim*1.

tvxconstraint = 1.0*nuxgrad*truetvx

# compute norm of Dy
xim = randn(nx,ny)
xim *= mask
npower = 50
for i in range(npower):
   yg = grady(xim)
   xim.fill(0.)
   xim  = mdivy(yg)
   xim *= mask
   xnorm2 = sqrt( (xim**2.).sum() )
   xim/=xnorm2
gnorm = sqrt(xnorm2)
nuygrad = nuyfact/gnorm
print("nuygrad: ",nuygrad)

tvyconstraint = 1.0*nuygrad*truetvy

sinodatasc = nusino*sinodata

# compute total norm
xim = randn(nx,ny)
xim *= mask
xim1 = xim*0.
xim2 = xim*0.
npower = 200
worksino =truesino*0.
for i in range(npower):
   circularFanbeamProjection(xim,worksino)
   worksino = fo(worksino)
   worksino *= nusino
   xg = gradx(xim)
   xg *= nuxgrad

   yg = grady(xim)
   yg *= nuygrad

   yim = l1f*xim
   mag1 = sqrt((yim**2).sum() + (yg**2).sum() + (xg**2).sum() +(worksino**2).sum())
   yim /=mag1
   yg /=mag1
   xg /=mag1
   worksino/=mag1

   xim1.fill(0.)
   worksino = fo(worksino)
   circularFanbeamBackProjection(worksino,xim1)
   xim1 *= (nusino*mask)
   xim2  = mdivx(xg)
   xim2 *= (nuxgrad*mask)
   xim3  = mdivy(yg)
   xim3 *= (nuygrad*mask)
   xim =  xim1+xim2+xim3 +l1f*yim
   mag2 = sqrt( (xim**2.).sum() )
   xim/=mag2
   xim1.fill(0.)
   xim2.fill(0.)
   xim3.fill(0.)

   print(i, "mag1: ", mag1," mag2: ",mag2)

totalnorm = (mag1+mag2)*0.5
print("totalnorm: ",totalnorm)

#totalvec = xim*1.
#input("hi")

sig = stepbalance/totalnorm
tau = 1./(totalnorm*stepbalance)

theta = 1.0
itermax = 2000


itr = 0

xim.fill(0.)
yim = xim*0.
xbarim = xim*0.
wimp = xim*0.
wimq = xim*0.
ysino = sinogram*0.
ygradx = testimage*0.
ygrady = testimage*0.
derrs = []
ierrs = []
tvxs = []
tvys = []
tvs = []
istops = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,3000,4000,5000]
while itr< itermax:
   itr += 1

   ysinoold = ysino*1.
   ygradxold = ygradx*1.
   ygradyold = ygrady*1.
   yimold = yim*1.

   wimp.fill(0.)
   worksino = fo(ysino)
   circularFanbeamBackProjection(worksino,wimp)
   wimp *= nusino
   wimp *= mask
   wimqx = mdivx(ygradx)
   wimqx *= nuxgrad
   wimqx *= mask
   wimqy = mdivy(ygrady)
   wimqy *= nuygrad
   wimqy *= mask
   wiml1 = l1f*yim

   ximold = xim*1.
   xim = xim -tau*(wimp + wimqx + wimqy + wiml1)
   xim[xim<0.0] = 0.

   xbarim = xim + theta*(xim-ximold)

#data fidelity block
   worksino.fill(0.)
   circularFanbeamProjection(xbarim,worksino)
   worksino = fo(worksino)
   worksino *= nusino
   resid = worksino - sinodatasc
   wdist = sqrt((resid**2).sum())
   wdistn = (wdist/nusino)/sqrt(1.*nviews*nbins)
   derrs.append(wdistn)
   ysino = (ysino + sig*resid)
   ymag = sqrt( (ysino**2).sum() )
   if ymag - sig*nusino*epssc > 0:
      ysino *= (ymag-sig*nusino*epssc)/ymag
   else:
      ysino *= 0.

   tgx = gradx(xbarim)
   tgx *= nuxgrad
   tvc = sqrt((tgx**2)).sum()/nuxgrad
   tvxs.append(tvc)
   ptilx= ygradx + sig*tgx
   ptilmag = maximum(sqrt(ptilx**2 ),(2.-alpha))
   ygradx = (2.-alpha)*ptilx/ptilmag

   tgy = grady(xbarim)
   tgy *= nuygrad
   tvc = sqrt((tgy**2)).sum()/nuygrad
   tvys.append(tvc)
   ptily= ygrady + sig*tgy
   ptilmag = maximum(sqrt(ptily**2 ),(alpha))
   ygrady = alpha*ptily/ptilmag

   tl1 = l1f*xbarim
   ptil1= yim + sig*tl1
   ptilmag = maximum(sqrt(ptil1**2 ),(beta))
   yim = beta*ptil1/maximum(ptilmag,1.e-10)



#TVvals
   tgx,tgy = gradim(xbarim)
   tvc = sqrt((tgx**2 + tgy**2)).sum()
   tvs.append(tvc)



   ygradx = ygradxold - rho*(ygradxold - ygradx)
   ygrady = ygradyold - rho*(ygradyold - ygrady)
   ysino = ysinoold - rho*(ysinoold - ysino)
   yim = yimold - rho*(yimold - yim)
   xim = ximold - rho*(ximold - xim)

   idist = sqrt( ((xbarim-phimage)**2).sum()/ (nx*ny) )
   ierrs.append(idist)
   if itr in istops:
      print("Iter: ",itr," TVX: ",tvxs[-1]," : ",truetvx," TVY: ",tvys[-1]," : ",truetvy," TV: ",tvs[-1]," : ",truetv)
      print("derr: ",derrs[-1], " ierr: ",ierrs[-1])

np.save(resultsfile+"imageDTV_L1.npy",xim)
