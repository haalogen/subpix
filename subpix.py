# TODO: check with analytical derivatives
import sys
import numpy as np
import plot as pl

np.set_printoptions(precision=4)

# get Ny, Nx from args
Ny = int(sys.argv[1])
Nx = int(sys.argv[2])


"""
    create and fill f = lImg (Ny, Nx), g = rImg (Ny, Nx) :
    i, j = [0 .. Ny-1], [0 .. Nx-1]
    f = F(x + dx, y + dy)
    g = k*F(x, y) + b
"""
lImg = np.zeros((Ny, Nx))
rImg = np.zeros((Ny, Nx))

# choose intervals
absX = 1
absY = 1
del_x = 2.0*absX/(Nx-1) # X interval between heighbouring points
del_y = 2.0*absY/(Ny-1) # Y interval between heighbouring points


# get origK, origB, origDx, origDy
origK = 1.0
origB = 0.0
origDx = (absX * 2.0) / (Nx-1) / 2
origDy = (absY * 2.0) / (Ny-1) / 4


lX = np.linspace(-absX + origDx, absX + origDx, Nx)
lY = np.linspace(-absY + origDy, absY + origDy, Ny)

rX = np.linspace(-absX, absX, Nx)
rY = np.linspace(-absY, absY, Ny)


print 'lX: ', lX
print 'lY: ', lY, '\n'
print 'rX: ', rX
print 'rY: ', rY, '\n'

lXX, lYY = np.meshgrid(lX, lY)
rXX, rYY = np.meshgrid(rX, rY)


lImg = np.e ** (-lXX**2 -lYY**2)
rImg = origK * np.e ** (-rXX**2 - rYY**2) + origB

# Analytiacal derivatives of f (to compare with finite)
lImg_der_x = -2 * lImg * (rXX)
lImg_der_y = -2 * lImg * (rYY)

# Finite derivatives of f
lImg_finite_der_x = np.zeros(lXX.shape)
lImg_finite_der_y = np.zeros(lYY.shape)


#print 'lImg: \n', lImg, '\n'
#print 'rImg: \n', rImg, '\n'

#print 'lImg_der_x: \n', lImg_der_x, '\n'
#print 'lImg_der_y: \n', lImg_der_y, '\n'
#pl.plot_pictures(lImg, rImg)

"""
    S = (Ny-2) * (Nx-2)  <-- We cut out boundary points beacuse
    can't calculate (estimate) derivatives f'_y, f'_x in them.

    create matrix A (S, 4)
    s = 0 .. S;     s = (j-1) + (i-1) * (Nx-2),
    where i, j = [1 .. Ny-2], [1 .. Nx-2]

    fill matrix A:

    g:
    A[:, 0] = g[1, 1] .. g[1, Nx-2], g[2, 1] .. g[2, Nx-2] ...
             ... g[Ny-2, 1] .. g[Ny-2, Nx-2]

    1:
    A[:, 1] = 1 ... 1


    f'_x:
    2 * A[:, 2] = f[1, 2] - f[1, 0] .. f[1, Nx-1] - f[1, Nx-3] ...
        ... f[Ny-2, 2] - f[Ny-2, 0] .. f[Ny-2, Nx-1] - f[Ny-1, Nx-3]


    f'_y:
    2 * A[:, 3] = f[2, 1] - f[0, 1] .. f[Ny-1, 1] - f[Ny-3, 1] ...
        ... f[2, Nx-2] - f[0, Nx-2] .. f[Ny-1, Nx-2] - f[Ny-3, Nx-2]
"""
sLen = (Ny) * (Nx)
A = np.zeros((sLen, 4))

# vector (s, 1) 
# xi = f[1, 1] .. f[1, Nx-2] ... f[Ny-2, 1] .. f[Ny-2, Nx-2]
xi = np.zeros((sLen, 1))


diff_der_x = np.zeros(lImg_der_x.shape, dtype=np.float32)
diff_der_y = np.zeros(lImg_der_y.shape, dtype=np.float32)

for i in xrange(Ny): # i = [0; Ny)
    for j in xrange(Nx):  # j = [0; Nx)
        index_s = (j) + (i) * (Nx)
        A[index_s, 0] = rImg[i, j]
        A[index_s, 1] = 1
        
        # 1st accuracy order derivatives
        if j-0 < 1: # left x-boundary point
            lImg_finite_der_x[i, j] =  \
                                (lImg[i, j+1] - lImg[i, j]) / del_x
        elif (Nx-1)-j < 1: # right x-boundary point
            lImg_finite_der_x[i, j] =  \
                                (lImg[i, j] - lImg[i, j-1]) / del_x
        else: # not x-boundary point
            lImg_finite_der_x[i, j] =  \
                            0.5*(lImg[i, j+1] - lImg[i, j-1]) / del_x
        
        if i-0 < 1: # up y-boundary point
            lImg_finite_der_y[i, j] =  \
                                (lImg[i+1, j] - lImg[i, j]) / del_y
        elif (Ny-1)-i < 1: # down y-boundary point
            lImg_finite_der_y[i, j] =  \
                                (lImg[i, j] - lImg[i-1, j]) / del_y
        else: # not y-boundary point
            lImg_finite_der_y[i, j] =  \
                            0.5*(lImg[i+1, j] - lImg[i-1, j]) / del_y
        
        # Analytiacal derivatives
#        A[index_s, 2] = lImg_der_x[i, j]
#        A[index_s, 3] = lImg_der_y[i, j]
        
        # finite derivatives
        A[index_s, 2] = lImg_finite_der_x[i, j]
        A[index_s, 3] = lImg_finite_der_y[i, j]
        
#        diff_der_x[i, j] = A[index_s, 2] - lImg_der_x[i, j]
#        diff_der_y[i, j] = A[index_s, 3] - lImg_der_y[i, j]
        
        xi[index_s] = lImg[i, j]


# vector (4, 1) psi = k', b', dx, dy; where k' = 1/k, b' = -b/k
psi = np.zeros((4, 1))

# A * psi = xi  =>  psi = pinv(A) * xi  => k = 1\k', b = -k*b', dx,dy
A_pinv = np.linalg.pinv(A)
psi = np.dot(A_pinv, xi)

modelImg = psi[0] * rImg + psi[1]
#print 'SUM_i,j |lImg - modelImg|: ', sum(sum(abs(lImg - modelImg))), '\n'
#print 'SUM_i,j |lImg - rImg|: ', sum(sum(abs(lImg - rImg))), '\n'

print 'A * psi = xi \n'
print 'A: \n', A, '\n'
print 'xi: \n', xi, '\n'
print 'psi: \n', psi, '\n'

k = 1 / psi[0]
b = -k * psi[1]
dx = float(psi[2])
dy = float(psi[3])

print 'Original: k: %lf | b: %lf | dx: %r | dy: %r \n' % (origK, origB, origDx, origDy)
print 'k: %lf | b: %lf | dx: %r | dy: %r \n' % (k, b, dx, dy)

dx_pix = dx / (2*absX) * Nx
dy_pix = dy / (2*absY) * Ny
print 'dx_pix: %lf | dy_pix: % lf' % (dx_pix, dy_pix)


#with open('dx.txt', 'a') as f:
#    s = str(origDx) + ' ' + str(dx) + '\n'
#    f.write(s)
#    print "Writing to file copmleted."

pl.plt.show(block=True)
