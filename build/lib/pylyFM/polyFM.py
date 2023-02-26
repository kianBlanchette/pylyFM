import polyFM.utils as utils
import numpy as np
import multitaper.utils as mt
from modF import modf

class ModulatedF:

  def __init__(self,x,K,NW,P,alpha=0.75,nFFT='default',dpssIN=None,apIN=None,derivIN=None,plotEase=True):
    assert K > P, "Number of tapers needs to be larger than polynomial degree"

    self.N = len(x)
    self.x = x
    self.K = K
    self.NW = NW
    self.P = P
    self.alpha = alpha


    if nFFT == "default":
      self.nFFT = 2**np.ceil(np.log2(2*self.N))
    else:
      assert int(nFFT), "nFFT must be an integer"
      self.nFFT = nFFT

    assert self.nFFT >= self.N, "nFFT must be at least as large as the length of x"

    if dpssIN is None:
      dpssIN = mt.dpss(npts=N,nw=NW,kspec=K)

    if derivIN is None:
      derivIN = utils.dpss_deriv(V=dpssIN[0],ev=dpssIN[1],NW=NW)

    if apIN is None:
      apIN = utils.dpss_ap(V=dpssIN[0],maxdeg=P,alpha=alpha)
    
    self.dpssIN = dpssIN
    self.ev = ev
    self.derivIN = derivIN
    self.apIN = apIN

    del dpssIN, derivIN, apIN
    
    V = dpssIN.v
    H = self.apIN.H[:,1:]

    xEigenCoefs = compute_eigencoefficients(x, V, self.nFFT)

    instFreq = IF_compute(xEigenCoefs,V,self.derivIN)

    IFEigenCoefs = np.transpose(V) @ instFreq
    IFPolyEigenCoefs = np.transpose(H) @ IFEigenCoefs

    self.xEigenCoefs = xEigenCoefs
    self.instFreq = instFreq
    self.IFEigenCoefs = IFEigenCoefs
    self.IFPolyEigenCoefs = IFPolyEigenCoefs

    ModF = modf(IFPolyEigenCoefs,IFEigenCoefs,[P,K,nFFT])

    if plotEase:
      for j in range(P):
        idx = np.where(ModF[j,:] < 1)
        ModF[j,idx] = 1

    self.ModF = ModF