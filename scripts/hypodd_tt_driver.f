c     tt_driver -- hypoDD's OWN 1D forward model, exposed for the synthetic test.
c
c     Reproduces partials.f line for line: station-elevation handling
c     (top_adj(1)=0, top_adj(k)=top(k)+elv, source depth + elv), the
c     layer-boundary nudge, delaz2 distances, vs = v/ratio in single
c     precision, and ttime (fastest of refracted / direct).  Anything the
c     synthetic dt.ct is built from here is therefore, by construction,
c     what hypoDD will predict at the starting locations.
c
c     stdin :  nl
c              nl lines   top(km) vp(km/s) ratio
c              n
c              n lines    src_lat src_lon src_dep(km) sta_lat sta_lon sta_elv(km)
c     stdout:  n lines    tp ts dist(km)
c
c     build: see scripts/51_synthetic_dtct.py (build_driver)
      program ttdriver
      implicit none
      include 'hypoDD.inc'
      integer nl, n, i, k
      real v(MAXLAY), top(MAXLAY), ratio(MAXLAY), vs(MAXLAY)
      real top_adj(MAXLAY)
      doubleprecision slat, slon
      real sdep, blat, blon, elv, del, dist, az, tp, ts, ain
      read(*,*) nl
      do i=1,nl
         read(*,*) top(i), v(i), ratio(i)
         vs(i) = v(i)/ratio(i)
      enddo
      read(*,*) n
      do i=1,n
         read(*,*) slat, slon, sdep, blat, blon, elv
         do k=1,nl
            if (abs(sdep-top(k)).lt.0.0001) sdep = sdep-0.001
         enddo
         top_adj(1) = 0.0
         do k=2,nl
            top_adj(k) = top(k)+elv
         enddo
         call delaz2(slat, slon, blat, blon, del, dist, az)
         call ttime(dist, sdep+elv, nl, v, top_adj, tp, ain)
         call ttime(dist, sdep+elv, nl, vs, top_adj, ts, ain)
         write(*,'(3f12.5)') tp, ts, dist
      enddo
      end
