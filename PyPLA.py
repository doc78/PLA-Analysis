import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import glob


def process_blue(img,bShow=False,bSave=False,strSaveFilename='', picname=''):
    b,g,r = cv2.split(img)
    rgb = cv2.merge([r,g,b])

    ret, thresh = cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    figsize=(9,6)
    if bSave:
        figsize=(15, 10)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=figsize, sharex=True, sharey=True)

    ax = axes.ravel()

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(rgb,markers)
    rgb_seg=np.copy(rgb);
    rgb_seg[markers == -1] = [0,255,0]


    ax[0].imshow(rgb)
    ax[0].set_title('Original Image')
    ax[1].imshow(b, cmap=plt.cm.gray)
    ax[1].set_title('Blue component')
    ax[2].imshow(thresh, cmap=plt.cm.gray)
    ax[2].set_title('Otsu Threshold')
    ax[3].imshow(sure_bg, cmap=plt.cm.gray.reversed())
    ax[3].set_title('Background')
    ax[4].imshow(sure_fg, cmap=plt.cm.gray)
    ax[4].set_title('Foreground seeds')
    ax[5].imshow(unknown, cmap=plt.cm.gray)
    ax[5].set_title('Borders')
    ax[6].imshow(dist_transform, cmap=plt.cm.gray)
    ax[6].set_title('Distance Transform')
    ax[7].imshow(rgb_seg)
    ax[7].set_title('Final Segmentation (nr Obj=' + str(ret-1) + ')')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    fig.suptitle(picname + ' - Blue Circles', fontsize=16)
    if bSave:
        plt.savefig(strSaveFilename, dpi=300,pad_inches=0.1)
    if bShow:
        plt.show()
    return ret-1

def process_red(img,bShow=False,bSave=False,strSaveFilename='', picname=''):
    b,g,r = cv2.split(img)
    rgb = cv2.merge([r,g,b])

    r_denoised=cv2.blur(r,(2,2))
    r_blurred=cv2.blur(r,(5,5))
    diff=r_denoised.astype(float)-r_blurred.astype(float)
    diff[diff<0]=0
    diff=np.uint8(diff)
    ret, thresh = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)

    figsize=(9,6)
    if bSave:
        figsize=(15, 10)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()

    # connect closer dots
    kernel = np.ones((3,3),np.uint8)
    connected = cv2.dilate(thresh,kernel,iterations=3)
    connected=np.uint8(ndi.binary_fill_holes(connected))*255
    kernel = np.ones((3,3),np.uint8)
    img_erosion = cv2.erode(connected,kernel,iterations=2)
    unknown = cv2.subtract(connected,img_erosion)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(connected)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(rgb,markers)
    rgb_seg=np.copy(rgb);
    rgb_seg[markers == -1] = [0,255,0]


    #ax[0].imshow(r_denoised, cmap=plt.cm.gray)
    ax[0].imshow(rgb)
    ax[0].set_title('Original Image')
    ax[1].imshow(r, cmap=plt.cm.gray)
    ax[1].set_title('Red component')
    ax[2].imshow(r_blurred, cmap=plt.cm.gray)
    ax[2].set_title('Red Blurred')
    ax[3].imshow(diff, cmap=plt.cm.gray)
    ax[3].set_title('Difference')
    ax[4].imshow(thresh, cmap=plt.cm.gray)
    ax[4].set_title('Segmentation')
    ax[5].imshow(connected, cmap=plt.cm.gray)
    ax[5].set_title('Dilation')
    ax[6].imshow(unknown, cmap=plt.cm.gray)
    ax[6].set_title('Borders')
    ax[7].imshow(rgb_seg)
    ax[7].set_title('Final Segmentation (nr Obj=' + str(ret-1) + ')')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    fig.suptitle(picname + ' - Red Dots', fontsize=16)
    if bSave:
        plt.savefig(strSaveFilename, dpi=300,pad_inches=0.1)
    if bShow:
        plt.show()
    return ret-1

if __name__ == '__main__':
    bShow=False
    bSave=True
    strFolder="Unlabelled\\"
    allValues = []
    allValues.append(('Filename','Num Blue Circles','Num Red Dots'))
    for file in glob.glob(strFolder + "*.jpg"):
        img = cv2.imread(file,cv2.IMREAD_COLOR)
        numBlues=process_blue(img,bShow,bSave,strSaveFilename='figBlue_' + file[len(strFolder):-4] + '.jpg', picname=file[len(strFolder):-4])
        numReds=process_red(img,bShow,bSave,strSaveFilename='figRed_' + file[len(strFolder):-4] + '.jpg', picname=file[len(strFolder):-4])
        allValues.append((file[len(strFolder):],numBlues,numReds))

    np.savetxt('PLA_analysis.csv', allValues, delimiter=",", fmt="%s")