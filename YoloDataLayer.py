import os,sys
import caffe
import numpy as np
import cv2
import leveldb
import random
from numpy import dtype

def computeIOU(x1,x2,x3,x4,y1,y2,y3,y4):
    left=max([x1,x3])
    right=min([x2,x4])
    top=max([y1,y3])
    bottom=min([y2,y4])
    overlappedArea=(right-left)*(bottom-top)
    unionArea=(x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-overlappedArea
    iou=0
    if unionArea>0:
        iou=overlappedArea/unionArea
    return iou

def getAnnotationnInfo(src,engine='coco'):
    clsIdx={}
    count=0
    db=leveldb.LevelDB(src)
    for key,value in db.RangeIter():
        fields=value.split(';')
        cls_id=int(fields[len(fields)-1])
        if cls_id not in clsIdx.keys():
            clsIdx[cls_id]=1
        count+=1
    return count,len(clsIdx.keys())

class YoloDataLayer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom)>0:
            raise Exception('Data layer not support any input')
        
        if len(top)!=2:
            raise Exception('Yolo Data Layer must given 2 output{image,anchor}')
        
        params=eval(self.param_str)
        self.batch_size=1
        self.width=227
        self.height=227
        self.annotation_path=''
        self.image_path=''
        self.index=0
        self.total=0
        self.version=1
        self.anchorSize=0
        self.grid=1
        self.total_class=2
        self.globalCount=0
        self.nBox=2
        self.shuffle=False
        self.newMap=[]
        
        if 'batch_size' in params:
            self.batch_size=int(params['batch_size'])
        if 'size' in params:
            self.width=int(params['size'])
            self.height=self.width
        if 'width' in params:
            self.width=int(params['width'])
        if 'height' in params:
            self.height=int(params['height'])
        if 'total' in params:
            self.total=int(params['total'])
        if 'version' in params:
            self.version=int(params['version'])
        if 'grid_size' in params:
            self.grid=int(params['grid_size'])
        if 'total_class' in params:
            self.total_class=int(params['total_class'])
        if 'engine' in params:
            self.engine=params['engine']
        if 'bbox' in params:
            self.nBox=int(params['bbox'])
        if 'image_path' in params:
            self.image_path=params['image_path']
        if 'annotation_path' in params:
            self.annotation_path=params['annotation_path']
            
        if self.total==0:
            self.total,self.total_class=getAnnotationnInfo(self.annotation_path, self.engine)
        
        if 'shuffle' in params and int(params['shuffle'])==1:
            self.shuffle=True
            self.newMap=range(self.total)
            random.shuffle(self.newMap)
        
        if self.version>3 or self.version<1:
            raise Exception('Yolo Data Layer Only support version 1, 2 and 3')
        elif self.version==1:
            self.anchorSize=self.grid*self.grid*(5*self.nBox+self.total_class)
        elif self.version==2:
            self.anchorSize=self.grid*self.grid*5*self.nBox*self.total_class
            raise Exception('Not implement for version 2 yet')
        elif self.version==3:
            raise Exception('Not implement for version 3 yet')
    
    def reshape(self,bottom,top):
        top[0].reshape(*[self.batch_size,3,self.height,self.width])
        top[1].reshape(*[self.batch_size,self.anchorSize])
    
    def forward(self,bottom,top):
        if self.version==1:
            self.annotDB=leveldb.LevelDB(self.annotation_path)
            
            iW=np.floor(self.width/self.grid)
            iH=np.floor(self.height/self.grid)
            data_len=self.grid*self.grid*((5*self.nBox)+self.total_class)
            cell_stride=5*self.nBox
            grid_stride=(5*self.nBox)+self.total_class
            minus=0
            i=0
            while i<self.batch_size:
            #for i in range(self.batch_size):
                key=self.globalCount
                if self.shuffle:
                    key=self.newMap[self.globalCount]
                record=self.annotDB.Get(str(key),default=None)
                if record is None:
                    raise Exception('Key Error on %d'%key)
                packet=record.split(';')
                packLen=len(packet)-1
                image_file=os.path.join(self.image_path,packet[0])
                #print image_file
                nObj=int((packLen-1)/5)
                clses=[]
                bboxes=[]
                todo=[]
                
                raw_img=cv2.imread(image_file)[:,:,(2,1,0)]
                img_data=cv2.resize(raw_img,(self.width,self.height))
                img_data=img_data/255.
                width_ratio=np.float(self.width)/raw_img.shape[1]
                height_ratio=np.float(self.height)/raw_img.shape[0]
                
                for n in range(nObj):
                    clses.append(packet[n*5+5])
                    x=packet[n*5+1]
                    y=packet[n*5+2]
                    w=packet[n*5+3]
                    h=packet[n*5+4]
                    bboxes.append([np.float(x)*width_ratio,np.float(y)*height_ratio,np.float(w)*width_ratio,np.float(h)*height_ratio])
                    todo.append(n)
                    
                img_data=np.swapaxes(np.swapaxes(img_data,0,2),1,2)
                data_pack=np.zeros(data_len,dtype=np.float)
                hasObj=False
                for gridY in range(self.grid):
                    for gridX in range(self.grid):
                        px=gridX*iW
                        py=gridY*iH
                        peX=px+iW
                        peY=py+iH
                        tobeRemove=-1
                        grid_idx=gridY*self.grid+gridX
                        
                        for ibox in range(self.nBox):
                            data_pack[grid_idx*grid_stride+ibox*5+0]=px
                            data_pack[grid_idx*grid_stride+ibox*5+1]=py
                            data_pack[grid_idx*grid_stride+ibox*5+2]=peX
                            data_pack[grid_idx*grid_stride+ibox*5+3]=peY
                        for idx in todo:
                            bbox=bboxes[idx]
                            cls=int(clses[idx])
                            iou=0
                            iou=computeIOU(px,bbox[0],peX,bbox[0]+bbox[2],py,bbox[1],peY,bbox[1]+bbox[3])
                            if iou>0 and data_pack[grid_idx*grid_stride+ibox*5+4]<iou:
                                for ibox in range(self.nBox):
                                    data_pack[grid_idx*grid_stride+ibox*5+0]=bbox[0]
                                    data_pack[grid_idx*grid_stride+ibox*5+1]=bbox[1]
                                    data_pack[grid_idx*grid_stride+ibox*5+2]=bbox[0]+bbox[2]
                                    data_pack[grid_idx*grid_stride+ibox*5+3]=bbox[1]+bbox[3]
                                data_pack[grid_idx*grid_stride+ibox*5+4]=iou
                                tobeRemove=idx
                                data_pack[grid_idx*grid_stride+cell_stride+cls]=1
                                #if packet[0]=='COCO_val2014_000000131159.jpg':
                                #    print bbox,
                                #    print ' ',
                                #    print grid_idx*grid_stride+cell_stride+cls,
                                #    print data_pack[grid_idx*grid_stride:grid_idx*grid_stride+cell_stride+self.total_class]
                                hasObj=True
                                
                            
                        if tobeRemove>-1:
                            todo.remove(tobeRemove)
                        tobeRemove=-1
                            
                #top[0].data[...]=
                if hasObj:
                    top[1].data[i,...]=data_pack
                    top[0].data[i,...]=img_data
                    i+=1
                else:
                    print 'skip'
                self.globalCount+=1
                if self.globalCount>=self.total:
                    self.globalCount=0
                    if self.shuffle:
                        random.shuffle(self.newMap)
        elif self.version==2:
            pass
        else:
            pass
    
    def backward(self,top,propagate_down,bottom):
        pass
