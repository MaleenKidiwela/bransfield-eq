c     path_driver -- hypoDD's simulps pseudo-bending tracer (ray_3d.f) in isolation.
c     stdin : model file name / lat lon rot / ipha ndip iskip scale1 scale2 xfac tlim nitpb
c             n / n lines: isp xe ye ze xr yr zr     (km, hypoDD frame, z down)
c     stdout: n lines: ttime az toa v_at_source
      program pathdriver
      implicit none
      character fn*110
      real lat, lon, rot, scale1, scale2, xfac, tlim
      integer ipha, ndip, iskip, nitpb, n, i, isp
      real xe, ye, ze, xr, yr, zr, tt, az, toa, vv
      read(*,'(a)') fn
      read(*,*) lat, lon, rot
      read(*,*) ipha, ndip, iskip, scale1, scale2, xfac, tlim, nitpb
      open(16, file='path_driver.log', status='unknown')
      call get_vel3d(fn, ipha, lat, lon, rot, ndip, iskip, scale1, scale2, xfac, tlim, nitpb, 16)
      read(*,*) n
      do i=1,n
         read(*,*) isp, xe, ye, ze, xr, yr, zr
         call path(isp, xe, ye, ze, xr, yr, zr, tt, az, toa, vv)
         write(*,'(4f12.5)') tt, az, toa, vv
      enddo
      end
