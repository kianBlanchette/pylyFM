function modFcomp(FPcoef, Fcoef, mxdeg, nord)

implicit none

integer, intent(in) :: mxdeg, nord
real*8, intent(in) :: FPcoef(mxdeg), Fcoef(nord)

integer :: k, L
real*8 :: ssq1, ssq2
real*8, dimension(mxdeg) :: Fp, modFcomp

ssq1 = 0.0d0
do k = 1, nord
    ssq1 = ssq1 + Fcoef(k)**2
end do
	
ssq2 = 0.0d0
do L = 1, mxdeg
    ssq2 = ssq2 + FPcoef(L)**2
    Fp(L) = dble(nord-L) * FPcoef(L)**2 / (ssq1 - ssq2)
end do

modFcomp = Fp

end function modFcomp
