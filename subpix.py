import sys
import numpy as np
import plot as pl

# get Ny, Nx from args
Ny = int(sys.argv[1])
Nx = int(sys.argv[2])


# get origK, origB, origDx, origDy
origK = 1.0
origB = 0.25
origDx = -2.0 / (Nx-1) / 4
origDy = 2.0 / (Ny-1) / 4

"""
    create and fill f = lImg (Ny, Nx), g = rImg (Ny, Nx) :
    i, j = [0 .. Ny-1], [0 .. Nx-1]
    f = F(x + dx, y + dy)
    g = k*F(x, y) + b
"""
lImg = np.zeros((Ny, Nx))
rImg = np.zeros((Ny, Nx))

# choose intervals:[-1; 1] for example
absX = 1
absY = 1

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

print 'lImg: \n', lImg, '\n'
print 'rImg: \n', rImg, '\n'
pl.plot_pictures(lImg, rImg)

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
sLen = (Ny-2) * (Nx-2)
A = np.zeros((sLen, 4))

# vector (s, 1) 
# xi = f[1, 1] .. f[1, Nx-2] ... f[Ny-2, 1] .. f[Ny-2, Nx-2]
xi = np.zeros((sLen, 1))


for i in xrange(1, Ny-1): # i = [1; Ny-1)
    for j in xrange(1, Nx-1):  # j = [1; Nx-1)
        index_s = (j-1) + (i-1) * (Nx-2)
        A[index_s, 0] = rImg[i, j]
        A[index_s, 1] = 1
        A[index_s, 2] = 0.5 * (lImg[i, j+1] - lImg[i, j-1])
        A[index_s, 3] = 0.5 * (lImg[i+1, j] - lImg[i-1, j])
        
        xi[index_s] = lImg[i, j]


# vector (4, 1) psi = k', b', dx, dy; where k' = 1/k, b' = -b/k
psi = np.zeros((4, 1))

# A * psi = xi  =>  psi = pinv(A) * xi  => k = 1\k', b = -k*b', dx,dy
A_pinv = np.linalg.pinv(A)
psi = np.dot(A_pinv, xi)

print 'A_pinv: \n', A_pinv, '\n'

print 'A * psi = xi \n'
print 'A: \n', A, '\n'
print 'xi: \n', xi, '\n'
print 'psi: \n', psi, '\n'

k = 1 / psi[0]
b = -k * psi[1]
dx = psi[2]
dy = psi[3]

print 'Original: k: %lf | b: %lf | dx: %lf | dy: %lf \n' % (origK, origB, origDx, origDy)
print 'k: %lf | b: %lf | dx: %lf | dy: %lf \n' % (k, b, dx, dy)

pl.plt.show(block=True)
