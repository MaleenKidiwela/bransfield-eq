c     proj_driver -- hypoDD's OWN short-distance projection (setorg/dist/redist),
c     exposed so the 3D model builder places nodes exactly where hypoDD will
c     look for them.  Frame (verified 2026-09-13): x positive EAST, y positive
c     NORTH, rot anticlockwise; km.
c     stdin :  orlat orlon rot
c              mode           (-1: lat lon -> x y ;  1: x y -> lat lon)
c              n
c              n lines of pairs
c     stdout:  n lines of pairs
      program projdriver
      implicit none
      real orlat, orlon, rot, x, y
      doubleprecision lat, lon
      integer mode, n, i
      read(*,*) orlat, orlon, rot
      call setorg(orlat, orlon, rot, 0)
      read(*,*) mode
      read(*,*) n
      do i=1,n
         if (mode.eq.-1) then
            read(*,*) lat, lon
            call dist(lat, lon, x, y)
            write(*,'(2f14.6)') x, y
         else
            read(*,*) x, y
            call redist(x, y, lat, lon)
            write(*,'(2f14.8)') lat, lon
         endif
      enddo
      end
