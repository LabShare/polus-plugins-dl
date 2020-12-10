'''

Most of the sourced  code is so from Cellpose repo  https://github.com/MouseLand/cellpose/tree/master/cellpose

'''

from scipy.ndimage.filters import maximum_filter1d
import scipy.ndimage
import numpy as np

from numba import njit
import utils


def flow_error(maski, dP_net):
    """ Error in flows from predicted masks vs flows predicted by network run on image.This function serves to benchmark the quality of masks, it works as follows.
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted
    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Args:
    maski(ND-array[int]): Masks produced from running dynamics on dP_net,
        where 0=NO masks; 1,2... are mask labels
    dP_net(ND-array[float]): ND flows where dP_net.shape[1:] = maski.shape

    Returns:
    flow_errors( array[float]):  float array with length maski.max().mean squared error between predicted flows and flows from masks.
    dP_masks (ND-array[float]): ND flows produced from the predicted masks

    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    # flows predicted from estimated masks
    dP_masks, _ = masks_to_flows(maski)
    iun = np.unique(maski)[1:]
    flow_errors = np.zeros((len(iun),))
    for i, iu in enumerate(iun):
        ii = maski == iu
        if dP_masks.shape[0] == 2:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii] / 5.) ** 2
                               + (dP_masks[1][ii] - dP_net[1][ii] / 5.) ** 2).mean()
        else:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii] / 5.) ** 2 * 0.5
                               + (dP_masks[1][ii] - dP_net[1][ii] / 5.) ** 2
                               + (dP_masks[2][ii] - dP_net[2][ii] / 5.) ** 2).mean()
    return flow_errors, dP_masks


@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ Run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)

    Args:
    T(array[float64]):  _ x Lx array that diffusion is run in
    y(array[int32]): pixels in y inside mask
    x(array[int32]): pixels in x inside mask
    ymed(int32):  center of mask in y
    xmed(int32): center of mask in x
    Lx(int32): size of x-dimension of masks
    niter(int32): number of iterations to run diffusion

    Returns:
    T(array[float64]): amount of diffused particles at each pixel

    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x] + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1] + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T



def masks_to_flows(masks):
    """ Convert masks to flows using diffusion from center pixel.Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside themask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Args:
    masks(array[int32]):  2D or 3D array.labelled masks 0=NO masks; 1,2,...=mask labels

    Returns:
    mu(array[float]):  3D or 4D array.flows in Y = mu[-2], flows in X = mu[-1].if masks are 3D, flows in Z = mu[0].
    mu_c(array[float]):  2D or 3D array.For each pixel, the distance to the center of the mask in which it resides

    """
    if masks.ndim > 2:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0,_ = masks_to_flows(masks[z])
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0,_ = masks_to_flows(masks[:,y])
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0,_ = masks_to_flows(masks[:,:,x])
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

@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ Run dynamics of pixels to recover masks in 3D.Euler integration of dynamics dP for niter steps.

    Args:
    p(array[float32]): pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
    dP(array[float]): flows [axis x Lz x Ly x Lx]
    inds(array[int32]): non-zero pixels to run dynamics on [npixels x 3]
    niter(int32): number of iterations of dynamics to run

    Returns:
    p(array[float32]):  final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):

        for j in range(inds.shape[0]):
            z = inds[j,0]
            y = inds[j,1]
            x = inds[j,2]
            p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
            p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] - dP[0,p0,p1,p2]))
            p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] - dP[1,p0,p1,p2]))
            p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] - dP[2,p0,p1,p2]))
    return p

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ Run dynamics of pixels to recover masks in 2D.
    Euler integration of dynamics dP for niter steps

    Args:
    p(array[float32]):  3D array.Pixel locations [axis x Ly x Lx] (start at initial meshgrid)
    dP(array[float32]): 3D array.flows [axis x Ly x Lx]
    inds(array[int32]):  2D array.non-zero pixels to run dynamics on [npixels x 2]
    niter(int32): Number of iterations of dynamics to run

    Returns:
    p(array[float32]):  3D array.Final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            p[0,y,x] = min(shape[0]-1, max(0, p[0,y,x] - dP[0,p0,p1]))
            p[1,y,x] = min(shape[1]-1, max(0, p[1,y,x] - dP[1,p0,p1]))
    return p


def follow_flows(dP, niter=200):
    """ define pixels and run dynamics to recover masks in 2D.Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

   Args:
    dP(array[float32]):  3D or 4D array .flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    niter(int):  default 200.Number of iterations of dynamics to run

    Returns:
    p(array[float32]): 3D array.Final locations of each pixel after dynamics

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)
    if len(shape)>2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)


        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        p = steps2D(p, dP, inds, niter)
    return p

def remove_bad_flow_masks(masks, flows, threshold=0.4):
    """ Remove masks which have inconsistent flows.
    Uses metrics.flow_error to compute flows from predicted masks
    and compare flows to predicted flows from network. Discards
    masks with flow errors greater than the threshold.

    Args:
    masks(array[int]):  2D or 3D array.Labelled masks, 0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]
    flows(array[float]):  3D or 4D array.flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    threshold(optional[float]):  default 0.4.Masks with flow error greater than threshold are discarded.

    Returns:
    masks(array[int]): Masks with inconsistent flow masks removed,0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]

    """
    merrors, _ = flow_error(masks, flows)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0

    return masks


def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4):
    """ create masks using pixel convergence after running dynamics

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Args:
    p(array[float32]):  3D or 4D array.Final locations of each pixel after dynamics,size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell(array[bool]):  2D or 3D array if iscell is not None, set pixels that are iscell False to stay in their original location.
    rpad(optional[int]):   default 20.histogram edge padding
    threshold(optional[float]):   default 0.4.Masks with flow error greater than threshold are discarded(if flows is not None)
    flows(optional[array] float):  3D or 4D array .flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows is not None, then masks with inconsistent flows are removed using
    `remove_bad_flow_masks`.

    Returns:
    M0(array[int]):  2D or 3D array.Masks with inconsistent flow masks removed,0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]

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


def fill_holes(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    Args:

    masks(array[int]): 2D or 3D array.Labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size(optional[int]):  default 15. Minimum number of pixels per mask
    Returns:
    masks(array[int]): int, 2D or 3D array.masks with holes filled and masks smaller than min_size removed,0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    slices = scipy.ndimage.find_objects(masks)
    i = 0
    for slc in slices:
        if slc is not None:
            msk = masks[slc] == (i+1)
            if msk.ndim==3:
                small_objects = np.zeros(msk.shape, np.bool)
                for k in range(msk.shape[0]):
                    msk[k] = scipy.ndimage.morphology.binary_fill_holes(msk[k])

            else:
                msk = scipy.ndimage.morphology.binary_fill_holes(msk)
                small_objects = ~remove_small_objects(msk, min_size=min_size)
            sm = np.logical_and(msk, small_objects)

            masks[slc][msk] = (i+1)
            masks[slc][sm] = 0
        i+=1
    return masks

def remove_small_objects(ar, min_size=64, connectivity=1):
    """ copied from skimage.morphology.remove_small_objects (required to be separate for pyinstaller) """
    out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = scipy.ndimage.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        scipy.ndimage.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")
    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out
