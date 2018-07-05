import os,sys,leveldb
import cv2
import numpy as np
import ujson

def computeIOU(x1,x2,x3,x4,y1,y2,y3,y4):
    left=max([x1,x3])
    right=min([x2,x4])
    top=max([y1,y3])
    bottom=min([y2,y4])
    overlappedArea=np.sqrt(pow(right-left,2)+pow(bottom-top,2))
    intersectArea=np.sqrt(pow(x2-x1,2)+pow(y2-y1,2))+np.sqrt(pow(x4-x3,2)+pow(y4-y3,2))-overlappedArea
    iou=overlappedArea/intersectArea
    return iou

#img_path=sys.argv[1]
annotation_path=sys.argv[1]
db_name=sys.argv[2]
data_eng='coco'

if len(sys.argv)>3:
    data_eng=sys.argv[3]

dst_path='.'
if len(sys.argv)>4:
    dst_path=sys.argv[4]
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

data_db_name='%s.ldb'%db_name
index_name='%s_index.txt'%db_name
info_name='%s_info.txt'%db_name

indexMap={}
mapIndex={}
imageMap={}
ikeyCount=0

image_data={}

if data_eng=='coco':
    # build keys
    with open(annotation_path,'r') as f:
        annots=ujson.load(f)
        classes=annots['categories']
        print 'Building Classes Map...',
        with open(os.path.join(dst_path,index_name),'wb') as wf:
            for cls in classes:
                wf.write('%d:%d:%s\n'%(ikeyCount,int(cls['id']),cls['name']))
                indexMap[int(cls['id'])]=cls['name']
                if cls['name'] not in mapIndex.keys():
                    mapIndex[cls['name']]=ikeyCount
                ikeyCount+=1
                if ikeyCount%1000==0:
                    print 'added %d keys'%ikeyCount
        print '[Done]'
        print 'Building Image Map...',
        for img in annots['images']:
            imageMap[img['id']]=img['file_name']
            image_data[img['id']]=''
        print '[Done]'
        data_db=leveldb.LevelDB(os.path.join(dst_path,data_db_name))
        dataCount=0
        
        print 'Generating Image Package...',
        #print annots['images']
        for annot in annots['annotations']:
            box=annot['bbox']
            #data_str='%s;%f;%f;%f;%f;%d'%(imageMap[annot['image_id']],
            #                              np.float(box[0]),np.float(box[1]),
            #                                       np.float(box[2]),np.float(box[3]),
            #                              mapIndex[indexMap[annot['category_id']]])
            
            data_str='%f;%f;%f;%f;%d'%(np.float(box[0]),np.float(box[1]),
                                                   np.float(box[2]),np.float(box[3]),
                                          mapIndex[indexMap[annot['category_id']]])
            if imageMap[annot['image_id']]=='':
                imageMap[annot['image_id']]+=data_str
            else:
                imageMap[annot['image_id']]+=';'+data_str
            #data_db.Put(str(dataCount),data_str)
            #dataCount+=1
            #if dataCount%10000==0:
                #print '\rGenerating LevelDB...saved %d records'%dataCount,
        print '[Done]'
        print 'Generating LevelDB...',
        for key in imageMap.keys():
            data_db.Put(str(dataCount),'%s;%s'%(imageMap[key],image_data[key]))
            dataCount+=1
            if dataCount%10000==0:
                print '\rGenerating LevelDB...saved %d records'%dataCount, 
        with open(os.path.join(dst_path,info_name),'w') as f:
            f.write('Number of record:%d'%dataCount)
    #with open(annotation_path,'r') as f:
    #    objects=ijson.items(f,'')
    #data_db=leveldb.LevelDB(os.path.join(dst_path,data_db_name))
    #label_db=leveldb.LevelDB(os.path.join(dst_path,label_db_name))
elif data_eng=='voc':
    data_db=leveldb.LevelDB(os.path.join(dst_path,data_db_name))
    #label_db=leveldb.LevelDB(os.path.join(dst_path,label_db_name))
else:
    print 'the dataset format is not support yet..'
    
print 'finish'