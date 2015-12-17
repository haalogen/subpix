import sys
import numpy as np
import plot as pl
import finite_difference as finite

from PIL import Image, ImageDraw

np.set_printoptions(precision=4)


path_to_img = sys.argv[1]
img = Image.open(path_to_img)

width, height = img.size

#data = np.array(list(img.getdata()), dtype=int)
#data = data.reshape((height, width))
#print "Data was extracted."

# The bounding box: (left, upper, right, lower) pixel coordinates
b_fac = 2 # big factor == orig_img.width / l_img.width
l_box = (0, 0, width/b_fac, height/b_fac) # box for left image

rf = 8 # resize factor
xd = rf/8 # x displacement
yd = rf/8 # y displacement
r_box = (xd, yd, width/b_fac + xd, height/b_fac + yd) # box for right image


l_img = img.crop(l_box) # left cropped Image
r_img = img.crop(r_box) # right cropped Image

# Create instruments for drawing on l_img, r_img
l_draw = ImageDraw.Draw(l_img)
r_draw = ImageDraw.Draw(l_img)

# Load the pixel values from l_img, r_img
l_pix = l_img.load()
r_pix = r_img.load()

# Add different noise on l_img and r_img
factor = 16 # parameter of noise dispersal

b_w, b_h = l_img.size # Big piece size (l_img or r_img)

for i in xrange(b_w):
    for j in xrange(b_h):
        rand = np.random.randint(-factor, factor+1) # [-factor, factor]
        print 'i, j, rand:', i, j, rand
        a = l_pix[i, j][0] + rand
        b = l_pix[i, j][1] + rand
        c = l_pix[i, j][2] + rand
        if (a < 0):
            a = 0
        if (b < 0):
            b = 0
        if (c < 0):
            c = 0
        if (a > 255):
            a = 255
        if (b > 255):
            b = 255
        if (c > 255):
            c = 255
        l_draw.point((i, j), (a, b, c))

#l_img.show()
#r_img.show()


# Convert l_img, r_img to Grayscale
l_img = l_img.convert('L') # RGB -> [0..255]
r_img = r_img.convert('L')

l_img.show()
r_img.show()

print 'Big piece size:', r_img.size

w, h = l_img.size

new_sz = (w/rf, h/rf) # for resizing images
new_shape = (h/rf, w/rf) # for reshaping matrices

l_img = l_img.resize(new_sz) 
r_img = r_img.resize(new_sz)

#l_img.show()
#r_img.show()


print '%r times cropped piece size:' % rf, r_img.size

print "Cropped images were created and resized."


l_data = np.array(list(l_img.getdata()), dtype=int)
l_data = l_data.reshape(new_shape)
r_data = np.array(list(r_img.getdata()), dtype=int)
r_data = r_data.reshape(new_shape)

assert(l_data.shape == r_data.shape)

# get Ny, Nx from pixmap
Nx, Ny = l_img.size
print 'Nx, Ny:', Nx, Ny


# choose intervals
absX = 1
absY = 1
del_x = 2.0*absX/(Nx-1) # X interval between heighbouring points
del_y = 2.0*absY/(Ny-1) # Y interval between heighbouring points

print 'del_x, del_y:', del_x, del_y

# Finite derivatives of f
l_finite_der_x = np.zeros(l_data.shape)
l_finite_der_y = np.zeros(r_data.shape)


sLen = (Ny) * (Nx)
A = np.zeros((sLen, 4))

# vector (s, 1) 
# xi = f[1, 1] .. f[1, Nx-2] ... f[Ny-2, 1] .. f[Ny-2, Nx-2]
xi = np.zeros((sLen, 1))

for i in xrange(Ny): # i = [0; Ny)
    for j in xrange(Nx):  # j = [0; Nx)
        index_s = (j) + (i) * (Nx)
        A[index_s, 0] = r_data[i, j]
        A[index_s, 1] = 1
        
        # 1st accuracy order derivatives
        if j-0 < 1: # left x-boundary point
            l_finite_der_x[i, j] =  \
                            (l_data[i, j+1] - l_data[i, j]) / del_x
        elif (Nx-1)-j < 1: # right x-boundary point
            l_finite_der_x[i, j] =  \
                            (l_data[i, j] - l_data[i, j-1]) / del_x
        else: # not x-boundary point
            l_finite_der_x[i, j] =  \
                        0.5*(l_data[i, j+1] - l_data[i, j-1]) / del_x
        
        if i-0 < 1: # up y-boundary point
            l_finite_der_y[i, j] =  \
                            (l_data[i+1, j] - l_data[i, j]) / del_y
        elif (Ny-1)-i < 1: # down y-boundary point
            l_finite_der_y[i, j] =  \
                            (l_data[i, j] - l_data[i-1, j]) / del_y
        else: # not y-boundary point
            l_finite_der_y[i, j] =  \
                        0.5*(l_data[i+1, j] - l_data[i-1, j]) / del_y
                
        # finite derivatives
        A[index_s, 2] = l_finite_der_x[i, j]
        A[index_s, 3] = l_finite_der_y[i, j]
        
        xi[index_s] = l_data[i, j]


# vector (4, 1) psi = k', b', dx, dy; where k' = 1/k, b' = -b/k
psi = np.zeros((4, 1))

# A * psi = xi  =>  psi = pinv(A) * xi  => k = 1\k', b = -k*b', dx,dy
A_pinv = np.linalg.pinv(A)
psi = np.dot(A_pinv, xi)


print 'A * psi = xi \n'
print 'A: \n', A, '\n'
#print 'xi: \n', xi, '\n'
print 'psi: \n', psi, '\n'

k = 1 / psi[0]
b = -k * psi[1]
dx = float(psi[2])
dy = float(psi[3])


#print 'Original: k: %lf | b: %lf | dx: %r | dy: %r \n' % (origK, origB, origDx, origDy)
print 'k: %lf | b: %lf | dx: %r | dy: %r \n' % (k, b, dx, dy)

dx_pix = dx / (2*absX) * Nx
dy_pix = dy / (2*absY) * Ny
print 'Expected dx_pix: %lf | dy_pix: %lf' % (-1.0*xd/rf, -1.0*yd/rf)
print 'Calculated dx_pix: %lf | dy_pix: %lf' % (dx_pix, dy_pix)


