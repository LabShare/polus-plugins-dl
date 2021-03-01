
from scipy.ndimage.filters import maximum_filter1d
import scipy.ndimage
import numpy as np

from numba import njit
import utils, metrics
import torch

torch_GPU = torch.device('cuda')


@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)

    Args:
        T(array[float64]): _ x Lx array that diffusion is run in
        y(array[int32]): pixels in y inside mask
        x(array[int32]): pixels in x inside mask
        ymed(int32): center of mask in y
        xmed(int32): center of mask in x
        Lx(int32): size of x-dimension of masks
        niter(int32): number of iterations to run diffusion
    Returns:
        T(array[float64]): amount of diffused particles at each pixel

    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T




def masks_to_flows(masks):
    """ convert masks to flows using diffusion from center pixel.Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Args:
        masks(array[int]):  2D or 3D array.labelled masks 0=NO masks; 1,2,...=mask labels
    Returns:
        mu(array[float]): 3D or 4D array.flows in Y = mu[-2], flows in X = mu[-1].if masks are 3D, flows in Z = mu[0].
        mu_c(array[float]):  2D or 3D array.for each pixel, the distance to the center of the mask in which it resides

    """
    if masks.ndim > 2:
        Lz, Ly, Lx = masks.shap
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows(masks[z])[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows(masks[:,y])[0]

            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows(masks[:,:,x])[0]
            mu[[0,1], :, :, x] += mu0
        return mu, None

    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    
    nmask = masks.max()
    slices = scipy.ndimage.find_objects(masks)
    dia = utils.diameters(masks)[0]
    s2 = (.15 * dia)**2
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y,x = np.nonzero(masks[sr, sc] == (i+1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
            xmed = x[imin]
            ymed = y[imin]

            d2 = (x-xmed)**2 + (y-ymed)**2
            mu_c[sr.start+y-1, sc.start+x-1] = np.exp(-d2/s2)

            niter = 2*np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly+2)*(lx+2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), niter)
            T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])

            dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx = T[y*lx + x+1] - T[y*lx + x-1]
            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((dy,dx))

    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    return mu, mu_c

@njit(['(int16[:,:,:],float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:],float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    Args:
        I(array[float32]) : C x Ly x Lx
        yc(array[float32]) : ni new y coordinates
        xc(array[float32]) : ni new x coordinates
        Y (array[float32]): C x ni I sampled at (yc,xc)

    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1= min(Ly-1, yf+1)
        xf1= min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )

def steps2D_interp(p, dP, niter, use_gpu=False):
    shape = dP.shape[1:]

    if use_gpu :
        device = torch_GPU

        pt = torch.from_numpy(p[[1,0]].T).double().to(device)
        pt = pt.unsqueeze(0).unsqueeze(0)

        pt[:,:,:,0] = (pt[:,:,:,0]/(shape[1]-1)) # normalize to between  0 and 1
        pt[:,:,:,1] = (pt[:,:,:,1]/(shape[0]-1)) # normalize to between  0 and 1
        pt = pt*2-1                       # normalize niterto between -1 and 1
        im = torch.from_numpy(dP[[1,0]]).double().to(device)
        im = im.unsqueeze(0)
        for k in range(2):
            im[:,k,:,:] /= (shape[1-k]-1) / 2.

        for t in range(niter):
            dPt = torch.nn.functional.grid_sample(im, pt,align_corners=True)

            for k in range(2):
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] - dPt[:,k,:,:], -1., 1.)

        pt = (pt+1)*0.5
        pt[:,:,:,0] = pt[:,:,:,0] * (shape[1]-1)
        pt[:,:,:,1] = pt[:,:,:,1] * (shape[0]-1)

        x=pt[:,:,:,[1,0]].cpu().numpy()
        return x.squeeze().T
    else:
        dPt = np.zeros(p.shape, np.float32)
        for t in range(niter):
            map_coordinates(dP, p[0], p[1], dPt)
            p[0] = np.minimum(shape[0]-1, np.maximum(0, p[0] - dPt[0]))
            p[1] = np.minimum(shape[1]-1, np.maximum(0, p[1] - dPt[1]))
        return p



@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D.Euler integration of dynamics dP for niter steps
    Args:
        p(array[float32]):  3D array.pixel locations [axis x Ly x Lx] (start at initial meshgrid)
        dP(array[float32]):3D array.flows [axis x Ly x Lx]
        inds(array[int32]): 2D array.non-zero pixels to run dynamics on [npixels x 2]
        niter(int32): number of iterations of dynamics to run
    Returns:
        p(array[float32]): 3D array.final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            p[0,y,x] = min(shape[0]-1, max(0, p[0,y,x] - dP[0,p0,p1]))
            p[1,y,x] = min(shape[1]-1, max(0, p[1,y,x] - dP[1,p0,p1]))
    return p

def follow_flows(dP, niter=200, interp=True, use_gpu=False):
    """ define pixels and run dynamics to recover masks in 2D
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)
    Args:
        dP(float32):  3D or 4D array.flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
        niter(int): default 200.number of iterations of dynamics to run
        interp(bool): default True.interpolate during 2D dynamics  (in previous versions + paper it was False)
        use_gpu(bool): default False.use GPU to run interpolated dynamics (faster than CPU)
    Returns:
        p(array[float32]): 3D array.final locations of each pixel after dynamics

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    p = np.array(p).astype(np.float32)
    # run dynamics on subset of pixels
    inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
    # fixes code breaks when the shape is 1
    if inds.shape[0] ==1 :
        interp=False
    if not interp:
        p = steps2D(p, dP, inds, niter)
    else:

         p[:,inds[:,0],inds[:,1]] = steps2D_interp(p[:,inds[:,0], inds[:,1]],
                                                      dP, niter, use_gpu=use_gpu)

    return p

def remove_bad_flow_masks(masks, flows, threshold=0.4):
    """ Remove masks which have inconsistent flows.Uses metrics.flow_error to compute flows from predicted masks and
    compare flows to predicted flows from network. Discards masks with flow errors greater than the threshold.
    Args:
        masks(array[int]): labelled masks, 0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]
        flows(array[float]): 3D or 4D array.flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
        threshold(float):  default 0.4.masks with flow error greater than threshold are discarded.
    Returns:
        masks(array[int]): int, 2D or 3D array masks with inconsistent flow masks removed,0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ = metrics.flow_error(masks, flows)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4):
    """ Create masks using pixel convergence after running dynamics.Makes a histogram of final pixel locations p, initializes
    masks at peaks of histogram and extends the masks from the peaks so that they include all pixels with more than 2 final pixels p.
    Discards masks with flow errors greater than the threshold.
    Args:
        p(array[float32]):final locations of each pixel after dynamics,size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        iscell(array[bool]): if iscell is not None, set pixels that are iscell False to stay in their original location.
        rpad(int):  default 20.histogram edge padding
        threshold(float): default 0.4.masks with flow error greater than threshold are discarded (if flows is not None)
        flows(array[float]): 3D or 4D array.flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows is not None, then masks
        with inconsistent flows are removed using`remove_bad_flow_masks`.
    Returns:
        M0(array[int]):masks with inconsistent flow masks removed, 0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]
    
    # remove big masks
    _,counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0==i] = 0
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if threshold is not None and threshold > 0 and flows is not None:
        M0 = remove_bad_flow_masks(M0, flows, threshold=threshold)
        _,M0 = np.unique(M0, return_inverse=True)
        M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0