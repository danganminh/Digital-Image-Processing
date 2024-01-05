import numpy as np


############################## RGB TO GRAYSCALE ########################################

def Grayscale_Lightness(img):
    R = np.array(img[:,:,0], dtype=np.float32)
    G = np.array(img[:,:,1], dtype=np.float32)
    B = np.array(img[:,:,2], dtype=np.float32)
    height, width = R.shape
    output = np.zeros([height, width], dtype=np.float32)
    for x in range(height):
        for y in range(width):
            min_value = np.min((R[x, y], G[x, y], B[x, y]))
            max_value = np.max((R[x, y], G[x, y], B[x, y]))
            output[x, y] = (min_value + max_value) / 2
    return output

def Grayscale_Average(img):
    R = np.array(img[:,:,0], dtype=np.float32)
    G = np.array(img[:,:,1], dtype=np.float32)
    B = np.array(img[:,:,2], dtype=np.float32)
    output = (R + G + B) / 3
    return output

def Grayscale_Luminosity(img):
    R = np.array(img[:,:,0], dtype=np.float32)
    G = np.array(img[:,:,1], dtype=np.float32)
    B = np.array(img[:,:,2], dtype=np.float32)
    output = 0.3*R + 0.59*G + 0.11*B
    return output

############################## RGB TO GRAYSCALE ########################################


############################## BLACK and WHITE ########################################

def Black_White(img):
    dim = img.ndim
    height, width = img.shape[0], img.shape[1]
    temp_matrix = np.ones([height, width])
    if dim == 3:
        temp_matrix = Grayscale_Average(img)
    if dim == 2:
        temp_matrix = img
    output = (temp_matrix / np.max(img)).round().astype(np.int32)
    return output

############################## BLACK and WHITE ########################################

############################## TRANSFORM IMAGE ########################################

def Rotate_Image(img, theta):
    angle = np.radians(theta)
    height, width = img.shape
    new_height = (np.abs(height*np.cos(angle)) + np.abs(width*np.sin(angle))).astype(np.int32)
    new_width = (np.abs(height*np.sin(angle)) + np.abs(width*np.cos(angle))).astype(np.int32)
    output = np.zeros([new_height, new_width], dtype=np.float32)
    cx = height // 2
    cy = width // 2
    center_x = new_height // 2
    center_y = new_width // 2
    for i in range(new_height):
        for j in range(new_width):
            x = (i - center_x)*np.cos(angle) - (j - center_y)*np.sin(angle)
            y = (i - center_x)*np.sin(angle) + (j - center_y)*np.cos(angle)
            x = round(x) + cx
            y = round(y) + cy
            if 0 <= x < height and 0 <= y < width:
                output[i, j] = img[x, y]
    return output

def Rotate_Image_3D(img, theta):
    angle = np.radians(theta)
    height, width, ndim = img.shape
    new_height = (np.abs(height*np.cos(angle)) + np.abs(width*np.sin(angle))).astype(np.int32)
    new_width = (np.abs(height*np.sin(angle)) + np.abs(width*np.cos(angle))).astype(np.int32)
    output = np.zeros([new_height, new_width, ndim], dtype=np.float32)
    for i in range(3):
        output[:,:,i] = Rotate_Image(img[:,:,i], theta)
    return output


def Translate_Image(img, tx, ty):
    height, width = img.shape
    new_height = (height + np.abs(tx)).astype(np.int32)
    new_width = (width + np.abs(tx)).astype(np.int32)
    output = np.zeros([new_height, new_width])
    for i in range(new_height):
        for j in range(new_width):
            x = (i - tx)
            y = (j - ty)
            if 0 <= x < height and 0 <= y < width:
                output[i, j] = img[x, y]
    return output

def Translate_Image_3D(img, tx, ty):
    height, width, ndim = img.shape
    output = np.zeros([height + np.abs(tx), width + np.abs(ty), ndim])
    for i in range(3):
        output[:,:,i] = Translate_Image(img[:,:,i], tx, ty)
    return output


def Scale_Image(img, fx, fy):
    height, width = img.shape
    new_height = round(height*fx)
    new_width = round(width*fy)
    output = np.zeros([new_height, new_width])
    for i in range(new_height):
        for j in range(new_width):
            x = round(i/fx)
            y = round(j/fy)
            if 0 <= x < height and 0 <= y < width:
                output[i, j] = img[x, y]
    return output

def Scale_Image_3D(img, fx, fy):
    height, width, ndim = img.shape
    output = np.zeros([height*fx, width*fy, ndim])
    for i in range(3):
        output[:,:,i] = Scale_Image(img[:,:,i], fx, fy)
    return output


def Shear_Image(img, sv, sh):
    height, width = img.shape
    new_height = (height + np.abs(height*sv)).astype(np.int32)
    new_width = (width + np.abs(width*sh)).astype(np.int32)
    output = np.zeros([new_height, new_width])
    for i in range(new_height):
        for j in range(new_width):
            x = round(i - sv*j)
            y = round(j - sh*i)
            if 0 <= x < height and 0 <= y < width:
                output[i, j] = img[x, y]
    return output

def Shear_Image_3D(img, sv, sh):
    height, width, ndim = img.shape
    new_height = (height + np.abs(height*sv)).astype(np.int32)
    new_width = (width + np.abs(width*sh)).astype(np.int32)
    output = np.zeros([new_height, new_width, ndim])
    for i in range(3):
        output[:,:,i] = Shear_Image(img[:,:,i], sv, sh)
    return output

def Subsample(img, factor):
    return img[::factor, ::factor]

def Subsample_3D(img, factor):
    return img[::factor, ::factor, :]


def Zero_Padding(img, nx, ny):
    height, width = img.shape
    height_new = height + 2*nx
    width_new = width + 2*ny
    output = np.zeros([height_new, width_new])
    output[nx:height_new-nx, ny:width_new-ny] = img
    return output


def Mirror_Padding(img, nx, ny):
    height, width = img.shape
    height_new = height + 2*nx
    width_new = width + 2*ny
    output = np.zeros([height_new, width_new])
    # Center
    output[nx:height_new-nx, ny:width_new-ny] = img
    # Top and Bottom
    output[:nx, ny:width_new-ny] = img[nx-1::-1, :] # Top
    output[height+nx:, ny:width_new-ny] = img[height-1:height-nx-1:-1, :] # Bottom
    # Left and Right
    output[:height_new, :ny] = output[:, 2*ny:ny:-1] # Left
    output[:height_new, width_new-ny:width_new] = output[:, width_new-ny-1:width_new-2*ny-1:-1] # Right
    return output

def Mirror_Padding_3D(img, nx, ny):
    height, width, ndim = img.shape
    height_new = height + 2*nx
    width_new = width + 2*ny
    output = np.zeros([height_new, width_new, ndim])
    for i in range(3):
        output[:,:,i] = Mirror_Padding(img[:,:,i], nx, ny)
    return output

############################## Histogram ########################################

def Hist_image(img):
    h = [(img==v).sum() for v in range(256)]
    hist = np.array(h) # Histogram
    norm = hist/hist.sum() # Normalized histogram
    return hist, norm

def cdf_img(img):
    cdf = np.zeros((256, ))
    _, norm = Hist_image(img)
    for i in range(256):
        cdf[i] = norm[i] + cdf[i-1]
    return cdf

def Hist_cdf_img_3D(img):
    hist_store = []
    norm_store = []
    cdf_strore = []
    for i in range(3):
        hist_value, norm_value = Hist_image(img[:,:,i].astype(np.float32))
        cdf_value = cdf_img(img[:,:,i].astype(np.float32))
        hist_store.append(hist_value)
        norm_store.append(norm_value)
        cdf_strore.append(cdf_value)
    return np.array(hist_store), np.array(norm_store), np.array(cdf_strore)

############################## Histogram ########################################

############################## Bit Plane ########################################

# Hàm chuyển từ số từ cơ số 10 sang cơ số nhị phân
def cov_binary(num):
    num_int = int(num.round())
    binary_num = [int(i) for i in list("{0:b}".format(num_int))]
    for j in range(8 - len(binary_num)):
        binary_num.insert(0, 0)
    return binary_num

# Hàm chuyển từ nhị phân sang cơ số 10
def conv_decimal(listt):
    x = 0
    for i in range(8):
        x = x + int(listt[i])*(2**(7-i))
    return x

# Hàm tách theo từng lớp bit khác nhau
def discriminate_bit(bit, img):
    m, n = img.shape
    z = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            x = cov_binary(img[i][j])
            for k in range(8):
                if k == bit:
                    x[k] = x[k]
                else:
                    x[k] = 0
            x1 = conv_decimal(x)
            z[i][j] = x1
    return z

def discriminate_bit_3D(bit, img):
    height, width, ndim = img.shape
    output = np.zeros([height, width, ndim], dtype=img.dtype)
    for i in range(3):
        output[:,:,i] = discriminate_bit(bit, img[:,:,i])
    return output

"""
    Using example: 
    array_img = []
    for i in range(8):
        array_img.append(discriminate_bit(i, dolar))
"""

############################## Bit Plane ########################################
