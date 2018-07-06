import os,sys
import caffe
import cv2
import numpy as np
from caffe import layers as L,params as P, to_proto

def genLayer(image_path,batch_size,img_size,grid_size,info,indexMap):
    total_class=0
    with open(indexMap,'r') as f:
        for ln in f:
            total_class+=1
    data_size=0
    with open(info,'r') as f:
        for ln in f:
            line=ln.rstrip('\n')
            keyvalue=line.split(':')
            if keyvalue[0]=='Number of record':
                data_size=int(keyvalue[1])
    param_str="'image_path':'%s','batch_size':%d,'size':%d,'grid_size':%d,'total_class':%d,'total':%d,'bbox:2"%(image_path,batch_size,img_size,grid_size,total_class,data_size)   
    dataLayer=L.Python(module='caffe.YoloDataLayer',layer='YoloDataLayer',param_str=param_str)
    print dataLayer
    to_proto(dataLayer)
    
def makePrototxt(filename,image_path,batch_size,img_size,grid_size,info,indexMap):
    with open(filename,'w') as f:
        print genLayer(image_path, batch_size, img_size, grid_size, info, indexMap)
        

caffe.set_mode_gpu()
net=caffe.Net(sys.argv[1],caffe.TEST)
net.forward()
img=net.blobs['data'].data
label=net.blobs['tag'].data
print label.shape
hasObj=False
for n in range(label.shape[0]):
    img1=np.swapaxes(np.swapaxes(img[n,:,:,:],0,2),0,1)[:,:,(2,1,0)].copy()
    hasObj=False
    for i in range(49):
        score=np.sum(label[n,i*90+10:i*90+90])
        if score>0.8:
            x=label[n,i*90]*img1.shape[1]
            y=label[n,i*90+1]*img1.shape[0]
            w=label[n,i*90+2]
            h=label[n,i*90+3]
            lx=int(np.floor(x-w*img1.shape[1]/2))
            ly=int(np.floor(y-h*img1.shape[0]/2))
            rx=int(np.floor(x+w*img1.shape[1]/2))
            ry=int(np.floor(y+h*img1.shape[0]/2))
            #print label[n,i*90:i*90+90],
            print score,
            cv2.rectangle(img1,(lx,ly),(rx,ry),(0,255,0),3)
            hasObj=True
    print ''
        #print label[n,i*90:i*90+90]
    if not hasObj:
        print 'object not found'
    cv2.imshow('preview',img1)
    cv2.waitKey()
print 'finished'