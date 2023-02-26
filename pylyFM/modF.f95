subroutine modF(Fp,IFPolyEigenCoefs, IFEigenCoefs, P, K, nFFT)
 	
implicit none

integer, intent(in) :: P, K, nFFT
real*8, intent(in) :: IFPolyEigenCoefs(P,nFFT), IFEigenCoefs(K,nFFT)
real*8, intent(out) :: Fp(P,K)

integer :: n

do n=1, nFFT
    Fp(:,n) = modFcomp(IFPolyEigenCoefs(:,n),IFEigenCoefs(:,n),P,K)
end do

contains

function modFcomp(IFPolyEigenCoefs, IFEigenCoefs, P, K)

implicit none

integer, intent(in) :: P, K
real*8, intent(in) :: IFPolyEigenCoefs(P), IFEigenCoefs(K)

integer  :: i, L
real*8 :: ssq1, ssq2
real*8, dimension(P) :: Fp, modFcomp

ssq1 = 0.0d0
do i = 1, K
    ssq1 = ssq1 + IFEigenCoefs(i)**2
end do
	
ssq2 = 0.0d0
do L = 1, P
    ssq2 = ssq2 + IFPolyEigenCoefs(L)**2
    Fp(L) = dble(K-L) * IFPolyEigenCoefs(L)**2 / (ssq1 - ssq2)
end do

modFcomp = Fp

end function modFcomp

end subroutine modF
