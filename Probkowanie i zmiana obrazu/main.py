import cv2
import matplotlib.pyplot as plt
import numpy as np

BIG1 = plt.imread('BIG_0001.jpg')
BIG2 = plt.imread('BIG_0002.jpg')
BIG3 = plt.imread('BIG_0003.jpg')
BIG4 = plt.imread('BIG_0004.png')
SMALL1 = plt.imread('SMALL_0001.tif')
SMALL2 = plt.imread('SMALL_0002.png')
SMALL3 = plt.imread('SMALL_0003.png')
SMALL4 = plt.imread('SMALL_0004.jpg')
SMALL5 = plt.imread('SMALL_0005.jpg')
SMALL6 = plt.imread('SMALL_0006.jpg')
SMALL7 = plt.imread('SMALL_0007.jpg')
SMALL8 = plt.imread('SMALL_0008.jpg')
SMALL9 = plt.imread('SMALL_0009.jpg')
SMALL10 = plt.imread('SMALL_0010.jpg')

def imgToFloat(img):
    if np.issubdtype(img.dtype, np.unsignedinteger):
        return img / 255.0
    elif np.issubdtype(img.dtype, np.floating):
        return img
    else:
        raise ValueError("Input image type not supported")

def nearestNeighbour(img, scale):
    img = imgToFloat(img)
    old_cols, old_rows, c = img.shape
    new_cols = np.ceil(old_cols * scale).astype(int)
    new_rows = np.ceil(old_rows * scale).astype(int)
    scaled_img = cv2.resize(img, (new_rows, new_cols))
    scaled_red = np.zeros((new_cols, new_rows))
    scaled_green = np.zeros((new_cols, new_rows))
    scaled_blue = np.zeros((new_cols, new_rows))
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    X = np.linspace(0, old_cols - 1, new_cols)
    Y = np.linspace(0, old_rows - 1, new_rows)
    for i in range(0, new_cols):
        for j in range(0, new_rows):
            old_col_idx = np.ceil(X[i]).astype(int)
            old_row_idx = np.ceil(Y[j]).astype(int)
            scaled_red[i, j] = r[old_col_idx, old_row_idx]
            scaled_green[i, j] = g[old_col_idx, old_row_idx]
            scaled_blue[i, j] = b[old_col_idx, old_row_idx]
    newIMG = np.stack((scaled_red, scaled_green, scaled_blue), axis=-1)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(newIMG)
    plt.title('Scaled')

    edges_original = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edges_scaled = cv2.Canny((scaled_img * 255).astype(np.uint8), 100, 200)

    plt.subplot(2, 2, 3)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Original Edges')

    plt.subplot(2, 2, 4)
    plt.imshow(edges_scaled, cmap='gray')
    plt.title('Scaled Edges')

    plt.show()

    return newIMG

def bilinearInterpolation(img, scale):
    img=imgToFloat(img)
    old_cols, old_rows, c = img.shape
    new_cols = np.ceil(old_cols * scale).astype(int)
    new_rows = np.ceil(old_rows * scale).astype(int)
    scaled_img = cv2.resize(img, (new_rows, new_cols))
    scaled_red=np.zeros((new_cols,new_rows))
    scaled_green=np.zeros((new_cols,new_rows))
    scaled_blue=np.zeros((new_cols,new_rows))
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    
    for i in range(0,new_cols):
        for j in range(0,new_rows):
            col_scaled = (i / new_cols) * (old_cols - 1)
            row_scaled = (j / new_rows) * (old_rows - 1)
            col_original=int(col_scaled)
            row_original=int(row_scaled)
            col_diff=col_scaled-col_original
            row_diff=row_scaled-row_original

            left_top_w = (1 - col_diff) * (1 - row_diff)
            right_top_w = col_diff * (1 - row_diff)
            left_bottom_w = (1 - col_diff) * row_diff
            right_bottom_w = col_diff * row_diff
            scaled_red[i, j] = left_top_w * r[col_original, row_original] + right_top_w * r[col_original + 1, row_original] + left_bottom_w * r[col_original, row_original + 1] + right_bottom_w * r[col_original + 1, row_original + 1]
            scaled_green[i, j] = left_top_w * g[col_original, row_original] + right_top_w * g[col_original + 1, row_original] + left_bottom_w * g[col_original, row_original + 1] + right_bottom_w * g[col_original + 1, row_original + 1]
            scaled_blue[i, j] = left_top_w * b[col_original, row_original] + right_top_w * b[col_original + 1, row_original] + left_bottom_w * b[col_original, row_original + 1] + right_bottom_w * b[col_original + 1, row_original + 1]
    newIMG=np.stack((scaled_red, scaled_green, scaled_blue), axis=-1)
    
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(newIMG)
    plt.title('Scaled')

    edges_original = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edges_scaled = cv2.Canny((scaled_img * 255).astype(np.uint8), 100, 200)

    plt.subplot(2, 2, 3)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Original Edges')

    plt.subplot(2, 2, 4)
    plt.imshow(edges_scaled, cmap='gray')
    plt.title('Scaled Edges')

    plt.show()
    
    return newIMG

def Mean(img,scale,static):
    img=imgToFloat(img)
    old_cols, old_rows, c = img.shape
    new_cols = np.ceil(old_cols * scale).astype(int)
    new_rows = np.ceil(old_rows * scale).astype(int)
    scaled_img = cv2.resize(img, (new_rows, new_cols))
    scaled_red=np.zeros((new_cols,new_rows))
    scaled_green=np.zeros((new_cols,new_rows))
    scaled_blue=np.zeros((new_cols,new_rows))
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    X=np.ceil(np.linspace(0,old_cols-1,new_cols))
    Y=np.linspace(0,old_rows-1,new_rows)
    for i in range(0,new_cols):
        if(static):
            ix = np.round(X[i] + np.arange(-3, 4)).astype(int).clip(0, old_cols - 1)
        else:
            if i>0:
                x1=-(X[i]-X[i-1])/2
            else:
                x1=0
            if i<len(X)-1:   
                x2=(X[i+1]-X[i])/2+1
            else:
                x2=0
            ix=np.round(X[i]+np.arange(x1,x2)).astype(int).clip(0,old_cols-1)
        
        for j in range(0,new_rows):
            if(static):
                iy = np.round(Y[j] + np.arange(-3, 4)).astype(int).clip(0, old_rows - 1)
            else:
                if j>0:
                    y1=-(Y[j]-Y[j-1])/2
                else:
                    y1=0
                if j<len(Y)-1:   
                    y2=(Y[j+1]-Y[j])/2+1
                else:
                    y2=0
                iy=np.round(Y[j]+np.arange(y1,y2)).astype(int).clip(0,old_rows-1)
            
            red_f= r[ix[0]:ix[-1],iy[0]:iy[-1]]
            green_f= g[ix[0]:ix[-1],iy[0]:iy[-1]]
            blue_f= b[ix[0]:ix[-1],iy[0]:iy[-1]]

            scaled_red[i, j] = np.mean(red_f)
            scaled_green[i, j] = np.mean(green_f)
            scaled_blue[i, j] = np.mean(blue_f)
               
    newIMG=np.stack((scaled_red, scaled_green, scaled_blue), axis=-1)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(newIMG)
    plt.title('Scaled')

    edges_original = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edges_scaled = cv2.Canny((scaled_img * 255).astype(np.uint8), 100, 200)

    plt.subplot(2, 2, 3)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Original Edges')

    plt.subplot(2, 2, 4)
    plt.imshow(edges_scaled, cmap='gray')
    plt.title('Scaled Edges')

    plt.show()

    return newIMG

def MeanWeight(img,scale,static):
    img=imgToFloat(img)
    old_cols, old_rows, c = img.shape
    new_cols = np.ceil(old_cols * scale).astype(int)
    new_rows = np.ceil(old_rows * scale).astype(int)
    scaled_img = cv2.resize(img, (new_rows, new_cols))
    scaled_red=np.zeros((new_cols,new_rows))
    scaled_green=np.zeros((new_cols,new_rows))
    scaled_blue=np.zeros((new_cols,new_rows))
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    X=np.ceil(np.linspace(0,old_cols-1,new_cols))
    Y=np.linspace(0,old_rows-1,new_rows)
    for i in range(0,new_cols):
        if(static):
            ix = np.round(X[i] + np.arange(-3, 4)).astype(int).clip(0, old_cols - 1)
        else:
            if i>0:
                x1=-(X[i]-X[i-1])/2
            else:
                x1=0
            if i<len(X)-1:   
                x2=(X[i+1]-X[i])/2+1
            else:
                x2=0
            ix=np.round(X[i]+np.arange(x1,x2)).astype(int).clip(0,old_cols-1)
        
        for j in range(0,new_rows):
            if(static):
                iy = np.round(Y[j] + np.arange(-3, 4)).astype(int).clip(0, old_rows - 1)
            else:
                if j>0:
                    y1=-(Y[j]-Y[j-1])/2
                else:
                    y1=0
                if j<len(Y)-1:   
                    y2=(Y[j+1]-Y[j])/2+1
                else:
                    y2=0
                iy=np.round(Y[j]+np.arange(y1,y2)).astype(int).clip(0,old_rows-1)
            
            red_f= r[ix[0]:ix[-1],iy[0]:iy[-1]]
            green_f= g[ix[0]:ix[-1],iy[0]:iy[-1]]
            blue_f= b[ix[0]:ix[-1],iy[0]:iy[-1]]

            rows, cols = red_f.shape
            wagi_wiersze = np.linspace(1, rows, rows).reshape(-1, 1)
            wagi_kolumny = np.linspace(1, cols, cols)
            wagi = np.outer(wagi_wiersze, wagi_kolumny)
            scaled_red[i, j] = np.average(red_f, weights=wagi)
            scaled_green[i, j] = np.average(green_f, weights=wagi)
            scaled_blue[i, j] = np.average(blue_f, weights=wagi)



                
    newIMG=np.stack((scaled_red, scaled_green, scaled_blue), axis=-1)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(newIMG)
    plt.title('Scaled')

    edges_original = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edges_scaled = cv2.Canny((scaled_img * 255).astype(np.uint8), 100, 200)

    plt.subplot(2, 2, 3)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Original Edges')

    plt.subplot(2, 2, 4)
    plt.imshow(edges_scaled, cmap='gray')
    plt.title('Scaled Edges')

    plt.show()

    return newIMG

def Median(img,scale,static):
    img=imgToFloat(img)
    old_cols, old_rows, c = img.shape
    new_cols = np.ceil(old_cols * scale).astype(int)
    new_rows = np.ceil(old_rows * scale).astype(int)
    scaled_img = cv2.resize(img, (new_rows, new_cols))
    scaled_red=np.zeros((new_cols,new_rows))
    scaled_green=np.zeros((new_cols,new_rows))
    scaled_blue=np.zeros((new_cols,new_rows))
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    X=np.ceil(np.linspace(0,old_cols-1,new_cols))
    Y=np.linspace(0,old_rows-1,new_rows)
    for i in range(0,new_cols):
        if(static):
            ix = np.round(X[i] + np.arange(-3, 4)).astype(int).clip(0, old_cols - 1)
        else:
            if i>0:
                x1=-(X[i]-X[i-1])/2
            else:
                x1=0
            if i<len(X)-1:   
                x2=(X[i+1]-X[i])/2+1
            else:
                x2=0
            ix=np.round(X[i]+np.arange(x1,x2)).astype(int).clip(0,old_cols-1)
        
        for j in range(0,new_rows):
            if(static):
                iy = np.round(Y[j] + np.arange(-3, 4)).astype(int).clip(0, old_rows - 1)
            else:
                if j>0:
                    y1=-(Y[j]-Y[j-1])/2
                else:
                    y1=0
                if j<len(Y)-1:   
                    y2=(Y[j+1]-Y[j])/2+1
                else:
                    y2=0
                iy=np.round(Y[j]+np.arange(y1,y2)).astype(int).clip(0,old_rows-1)
            
            red_f= r[ix[0]:ix[-1],iy[0]:iy[-1]]
            green_f= g[ix[0]:ix[-1],iy[0]:iy[-1]]
            blue_f= b[ix[0]:ix[-1],iy[0]:iy[-1]]

            scaled_red[i, j] = np.median(red_f)
            scaled_green[i, j] = np.median(green_f)
            scaled_blue[i, j] = np.median(blue_f)



                
    newIMG=np.stack((scaled_red, scaled_green, scaled_blue), axis=-1)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(newIMG)
    plt.title('Scaled')

    edges_original = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edges_scaled = cv2.Canny((scaled_img * 255).astype(np.uint8), 100, 200)

    plt.subplot(2, 2, 3)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Original Edges')

    plt.subplot(2, 2, 4)
    plt.imshow(edges_scaled, cmap='gray')
    plt.title('Scaled Edges')

    plt.show()
    

    return newIMG


# fragment = img[200:600, 1300:1700]

# x=nearestNeighbour(img, 5)
fragment = SMALL2[50:150, 30:90]
scaled_image = bilinearInterpolation(fragment, 6)