from scipy.fft import fft, ifft, fftn
from multitaper.utils import dpss
import numpy as np

####################################
# dpss_ap
"""
  computes a set of polynomials associated with the slepian sequences
  based on Chebyshev polynomials

  Inputs: V = matrix of slepian sequences
          maxdeg = highest degree polynomial to compute
          alpha = shape parameter of the chebyshev polynomial
  Returns: U = K x P matrix containing eigencoefficients of the associated polynomials (orthogonal columns)
           R = N x p matrix of the polynomials 
           Hn = sum of squares of the columns of R
"""
####################################
def dpss_ap(V,maxdeg,alpha=0.75):

  N, K = V.shape
  P = maxdeg+1

  R = np.zeros(shape=(N,P))
  U = np.zeros(shape=(K,P))

  #setup centered time index
  midTime = (N+1)/2
  scl = 2/(N-1)
  timeArrC = [(t - midTime) * scl for t in range(1,N+1)]

  R[:,0] = 1.0
  if maxdeg > 0:
    R[:,1] = [2*alpha*t for t in timeArrC]
    if maxdeg > 1:
      for j in range(2,P):
        A1 = 2 * (j - 1 + alpha) / j
        A2 = (j - 2 + 2 * alpha) / j
        R[:,j] = [A1 * t for t in timeArrC] * R[:,j-1] - A2 * R[:,j-2]

  #Inner products of R and V
  for L in range(P):
    Kmin = L % 2
    for k in [t for t in range(Kmin,K) if t % 2 == Kmin % 2]:
      U[k,L] = np.dot(V[:,k], R[:,L])

  #Degree 0, 1 (manual)
  for L in range(min(2,P)):
    scl = 1 / np.sqrt(sum( U[:,L]**2 ))
    U[:,L] = U[:,L] * scl
    R[:,L] = R[:,L] * scl

  # loop on higher degrees, applying Gram-Schmidt only on similar parity functions
  # as even/odd are already orthogonal in U
  if P > 2:
    for L in range(2,P):
      Kmin = L % 2
      for j in [t for t in range(Kmin,L) if t % 2 == Kmin % 2]:
        scl = np.dot(U[:,L], U[:,j])
        U[:,L] = U[:,L] - scl * U[:,j]
        R[:,L] = R[:,L] - scl * R[:,j]
      scl = 1 / np.sqrt(sum( U[:,L]**2 ))
      U[:,L] = U[:,L] * scl
      R[:,L] = R[:,L] * scl

  Hn = np.sum(R**2, axis=0)
  return U, R, Hn
##################################
# end dpss_ap
##################################

##################################
# dpss_deriv
"""
  computes derivatives of slepian sequences by exploting the toeplitz structure of the
  matrix multiplication

  Inputs: V = matrix of slepian sequences
          ev = array of eigenvalues of slepian sequences
          NW = time-bandwidth product

  Returns: VDeriv = matrix of derivatives of slepian sequences
"""
##################################
def dpss_deriv(V,ev,NW):
  N = V.shape[0]
  K = V.shape[1]
  W = NW / N
  rng = range(1,N)
  tn = [t for t in rng] 
  tnw = [2 * np.pi * W * t for t in rng]
  tn2 = [t**2 for t in rng]
  
  b = np.insert(np.sin(tnw) / (pi * tn2), 0, 0)
  a = np.insert(2 * W * np.cos(tnw) / tn, 0, 0)
  y = a - b

  VDeriv = np.zeros(shape=(N,K), dtype='float', order='C')
  x = np.concatenate((np.append(y,0), np.flip(-y[1:])))

  for k in range(K):
  	p = np.concatenate((V[:,k], np.repeat(0,N)))
  	h = fft(p) * fft(x)
  	tmp = np.real(ifft(h)[:N] / len(h)) / ev[k]
  	VDeriv[:,k] = np.reshape(tmp, (N,1))

  return VDeriv
###################################
# end dpss_deriv
###################################

###################################
# compute_eigencoefficients
"""
  computes the eigencoefficients of a time series, given a set of tapers

  Inputs: x = time series 
          V = matrix of tapers (typically slepians)
          nFFT = number of frequencies where the eigencoefficients are defined

  Returns: yk = eigencoefficients of x
"""
###################################

def compute_eigencoefficients(x,V,nFFT):
  N = len(x)
  assert nFFT >= N, "nFFT needs to be >= N"
  k = V.shape[1]

  x = np.transpose(np.reshape(k * x, (k,N)))
  padX = np.row_stack(x, np.zeros(nFFT-N,k))
  padV = np.row_stack((V, np.zeros((nFFT-N,k))))
  taperX = padV * padX

  yk = fftn(taperX,axes=0)

  return yk

###################################
# end compute_eigencoefficients
###################################

###################################
# IF_compute
"""
  computes the instantaneous frequency series for a time series

  Inputs: yk = the eigencoefficients of a times series x
          V = the matrix of tapers (typically slepians) used to derive the eigencoefficients
          VDeriv = the matrix of derivatives of the tapers

  Returns: phi = a matrix of instantaneous frequency series around each frequency for which the 
                 eigencoefficients were computed

"""
###################################

def IF_compute(yk,V,VDeriv):
  U = np.transpose(V) @ np.real(yk)
  W = np.transpose(V) @ np.imag(yk)
  Udot = np.transpose(VDeriv) @ np.real(yk)
  Wdot = np.transpose(VDeriv) @ np.imag(yk)

  num = U * Wdot - Udot * W 
  amp2 = U**2 + W**2

  phi = num / (2 * np.pi * amp2)

  return phi

###################################
# end IF_compute
###################################
