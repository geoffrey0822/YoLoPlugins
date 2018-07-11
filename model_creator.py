import os,sys,caffe
import numpy as np
from caffe import layers as L
from caffe import params as P

def exportModel(net,fname,extra_str='',isDeploy=False):
    with open(fname,'w')as f:
        if isDeploy:
            f.write(extra_str+'\n'+'layer {'+'layer {'.join(str(net.to_proto()).split('layer {')[2:]))
        else:
            f.write('%s%s\n'%(extra_str,net.to_proto()))
        f.close()
         
def addPooling(net,input,prefix,ksize,pad=0,stride=1,pool='MAX'):
    layer_name='pool%s'%prefix
    if pool == 'MAX':
        net.tops[layer_name]=L.Pooling(input,name=layer_name,
                                       pooling_param={'kernel_size':ksize,'pad':pad,'stride':stride,'pool':P.Pooling.MAX})
    elif pool=='AVE':
        net.tops[layer_name]=L.Pooling(input,name=layer_name,
                                       pooling_param={'kernel_size':ksize,'pad':pad,'stride':stride,'pool':P.Pooling.AVE})
    return net.tops[layer_name]

def addConvolution(net,input,prefix,ksize,noutput,pad=0,stride=1,lr_mults=(0,0),decays=(0,0),hasBias=True,hasReLU=True,slope=0):
    conv_name='conv%s'%prefix
    relu_name='relu%s'%prefix
    if hasBias:
        net.tops[conv_name]=L.Convolution(input,name=conv_name,
                                convolution_param={'kernel_size':ksize,'num_output':noutput,'pad':pad,'stride':stride},
                                param=[{'lr_mult':lr_mults[0],'decay_mult':decays[0]},
                                       {'lr_mult':lr_mults[1],'decay_mult':decays[1]}],weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant',value=0))
    else:
        net.tops[conv_name]=L.Convolution(input,name=conv_name,
                            convolution_param={'kernel_size':ksize,'num_output':noutput,'pad':pad,'stride':stride,'bias_term':False},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':decays[0]}],weight_filler=dict(type='xavier'))
    if hasReLU:
        if slope==0:
            net.tops[relu_name]=L.ReLU(net.tops[conv_name],name=relu_name,in_place=True)
        else:
            net.tops[relu_name]=L.ReLU(net.tops[conv_name],name=relu_name,in_place=True,
                                       relu_param={'negative_slope':slope})
    return net.tops[conv_name]
    
def addBatchNorm(net,input,prefix,hasBias=True,isTrain=True,lr_mult=(0,0),decay_mult=(0,0)):
    bn_name='bn%s'%prefix
    scale_name='scale%s'%prefix
    if isTrain:
        net.tops[bn_name]=L.BatchNorm(input,name=bn_name,
                                      batch_norm_param={'use_global_stats':False},
                                      param=[{'lr_mult':0},{'lr_mult':0},{'lr_mult':0}])
        net.tops[scale_name]=L.Scale(net.tops[bn_name],name=scale_name,
                                     scale_param={'bias_term':hasBias},
                                     param=[{'lr_mult':lr_mult[0],'decay_mult':decay_mult[0]},
                                             {'lr_mult':lr_mult[1],'decay_mult':decay_mult[1]}])
    else:
        net.tops[bn_name]=L.BatchNorm(input,name=bn_name,
                                  batch_norm_param={'use_global_stats':True})
    net.tops[scale_name]=L.Scale(net.tops[bn_name],name=scale_name,
                                 scale_param={'bias_term':hasBias})
    return net.tops[scale_name]

def addReLU(net,input,prefix,slope=0):
    relu_name='relu%s'%prefix 
    if slope!=0:
        net.tops[relu_name]=L.ReLU(input,name=relu_name,in_place=True)
    else:
        net.tops[relu_name]=L.ReLU(input,name=relu_name,in_place=True,
                                   relu_param={'negative_slope':slope})
    
def addFC(net,input,prefix,noutput,lr_mult=(0,0),decay=(0,0),hasReLU=False,slope=0):
    fc_name='fc%s'%prefix
    relu_name='relu%s'%prefix
    net.tops[fc_name]=L.InnerProduct(input,name=fc_name,
                                     inner_product_param={'num_output':noutput},
                                     param=[{'lr_mult':lr_mult[0],'decay_mult':decay[0]},
                                            {'lr_mult':lr_mult[1],'decay_mult':decay[1]}],
                                     weight_filler=dict(type='xavier'),
                                     bias_filler=dict(type='constant',value=0))
    if hasReLU:
        if slope==0:
            net.tops[relu_name]=L.ReLU(net.tops[fc_name],name=relu_name,in_place=True)
        else:
            net.tops[relu_name]=L.ReLU(net.tops[fc_name],name=relu_name,in_place=True,
                                       relu_param={'negative_slope':slope})
    return net.tops[fc_name]
    
def addDropout(net,input,prefix,dropout_ratio=0.1,phase_str=''):
    layer_name='dropout%s'%prefix
    if phase_str !='TEST' and phase_str !='TRAIN':
        net.tops[layer_name]=L.Dropout(input,name=layer_name,in_place=True,
                                       dropout_param={'dropout_ratio':dropout_ratio})
    else:
        phase=caffe.TRAIN
        if phase_str=='TEST':
            phase=caffe.TEST
        net.tops[layer_name]=L.Dropout(input,name=layer_name,in_place=True,
                                       dropout_param={'dropout_ratio':dropout_ratio},
                                       include=dict(phase=phase))
    
def addYoloDataLayer(net,prefix,batch_size=1,version=1,shuffle=0,img_size=256,grid_size=7,nBBox=2,img_path='',annot_path='',total_class=2,stage='',datasize=0):
    param_str="{'batch_size':%d,'shuffle':%d,'size':%d,'total':%d,'version':%d,'grid_size':%d,'total_class':%d,'bbox':%d,'image_path':'%s','annotation_path':'%s'}"%(batch_size,shuffle,img_size,datasize,version,grid_size,total_class,nBBox,img_path,annot_path)
    if stage!='TEST' and stage!='TRAIN':
        net.tops['data'],net.tops['tag']=L.Python(name=prefix,
                                                  python_param={'module':'caffe.YoloDataLayer','layer':'YoloDataLayer','param_str':param_str})
    else:
        phase=caffe.TRAIN
        if stage=='TEST':
            phase=caffe.TEST
        net.tops['data'],net.tops['tag']=L.Python(name=prefix,
                                                  python_param={'module':'caffe.YoloDataLayer','layer':'YoloDataLayer','param_str':param_str},
                                                  include=dict(phase=phase),ntop=2)
    return net.tops['data'],net.tops['tag']
    
def addYoloV1Loss(net,prefix,inputs,grid_size,total_class,nBBox,threshold=0.1,noObjScale=0.5,coordScale=5.,stage=''):
    if stage!='TEST' and stage!='TRAIN':
        net.tops[prefix]=L.YoloV1(*inputs,name=prefix,
                                  yolo_v1_param={'class_num':total_class,'box_num':nBBox,'grid_size':grid_size,'threshold':threshold,'gamma_noobj':noObjScale,'gamma_coord':coordScale})
    else:
        phase=caffe.TRAIN
        if stage=='TEST':
            phase=caffe.TEST
        net.tops[prefix]=L.YoloV1(*inputs,name=prefix,
                                  yolo_v1_param={'class_num':total_class,'box_num':nBBox,'grid_size':grid_size,'threshold':threshold,'gamma_noobj':noObjScale,'gamma_coord':coordScale},
                                  include=dict(phase=phase))
    return net.tops[prefix]
        
def addSoftmaxLoss(net,prefix,inputs,stage=''):
    if stage!='TEST' and stage!='TRAIN':
        net.tops[prefix]=L.SoftmaxWithLoss(*inputs,name=prefix)
    else:
        phase=caffe.TRAIN
        if stage=='TEST':
            phase=caffe.TEST
        net.tops[prefix]=L.SoftmaxWithLoss(*inputs,name=prefix,
                                           include=dict(phase=phase))
    return net.tops[prefix]
        
def addSoftmax(net,prefix,input):
    net.tops[prefix]=L.Softmax(input,name=prefix)
    
def addLRN(net,prefix,input,alpha=1,beta=5,local_size=5,acrossCh=True):
    layer_name='norm%s'
    if acrossCh:
        net.tops[layer_name]=L.LRN(input,name=layer_name,
                                   lrn_param={'alpha':alpha,'beta':beta,'local_size':local_size})
    else:
        net.tops[layer_name]=L.LRN(input,name=layer_name,
                                   lrn_param={'alpha':alpha,'beta':beta,'local_size':local_size,'norm_region':P.LRN.WITHIN_CHANNEL})
    return net.tops[layer_name]

def addPassThrough(net,inputs,prefix,outputName,operation='SUM'):
    layer_name='res%s'%prefix
    if operation=='SUM':
        net.tops[outputName]=L.Eltwise(*inputs,name=layer_name,
                                       eltwise_param={'operation':P.Eltwise.SUM})
    elif operation=='PROD':
        net.tops[outputName]=L.Eltwise(*inputs,name=layer_name,
                                       eltwise_param={'operation':P.Eltwise.PROD})
    elif operation=='MAX':
        net.tops[outputName]=L.Eltwise(*inputs,name=layer_name,
                                       eltwise_param={'operation':P.Eltwise.MAX})
    else:
        print 'not support this operation'
    return net.tops[outputName]

def get_YoLoV1_DarkNet19_trainval(fname,train_image_path,train_annot_path,val_image_path,val_annot_path,class_num,train_batch=1,val_batch=1,grid=7,bbox=2,datasize=0):
    final_output=5*bbox+class_num
    net=caffe.NetSpec()
    data,tag=addYoloDataLayer(net, 'train-data', train_batch , 1, 1, 224, grid, bbox, train_image_path, train_annot_path, class_num, stage='TRAIN', datasize=datasize)
    addYoloDataLayer(net, 'val-data', val_batch , 1, 1, 224, grid, bbox, val_image_path, val_annot_path, class_num, stage='TEST', datasize=datasize)
    
    conv1=addConvolution(net, data, '1', ksize=3, noutput=32, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn1=addBatchNorm(net,conv1, '1', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn1, '1',slope=0.1)
    pool1=addPooling(net, bn1, '1', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv2=addConvolution(net, pool1, '2', ksize=3, noutput=64, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn2=addBatchNorm(net,conv2, '2', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn2, '2',slope=0.1)
    pool2=addPooling(net, bn2, '2', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv3=addConvolution(net, pool2, '3', ksize=3, noutput=128, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn3=addBatchNorm(net,conv3, '3', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn3, '3',slope=0.1)
    
    conv4=addConvolution(net, bn3, '4', ksize=1, noutput=64, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn4=addBatchNorm(net,conv4, '4', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn4, '4',slope=0.1)
    
    conv5=addConvolution(net, bn4, '5', ksize=3, noutput=128, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn5=addBatchNorm(net,conv5, '5', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn5, '5',slope=0.1)
    pool5=addPooling(net, bn5, '5', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv6=addConvolution(net, pool5, '6', ksize=3, noutput=256, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn6=addBatchNorm(net,conv6, '6', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn6, '6',slope=0.1)
    
    conv7=addConvolution(net, bn6, '7', ksize=1, noutput=128, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn7=addBatchNorm(net,conv7, '7', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn7, '7',slope=0.1)
    
    conv8=addConvolution(net, bn7, '8', ksize=3, noutput=256, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn8=addBatchNorm(net,conv8, '8', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn8, '8',slope=0.1)
    pool8=addPooling(net, bn8, '8', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv9=addConvolution(net, pool8, '9', ksize=3, noutput=512, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn9=addBatchNorm(net,conv9, '9', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn9, '9',slope=0.1)
    
    conv10=addConvolution(net, bn9, '10', ksize=1, noutput=256, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn10=addBatchNorm(net,conv10, '10', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn10, '10',slope=0.1)
    
    conv11=addConvolution(net, bn10, '11', ksize=3, noutput=512, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn11=addBatchNorm(net,conv11, '11', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn11, '11',slope=0.1)
    
    conv12=addConvolution(net, bn11, '12', ksize=1, noutput=256, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn12=addBatchNorm(net,conv12, '12', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn12, '12',slope=0.1)
    
    conv13=addConvolution(net, bn12, '13', ksize=3, noutput=512, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn13=addBatchNorm(net,conv13, '13', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn13, '13',slope=0.1)
    pool13=addPooling(net, bn13, '13', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv14=addConvolution(net, pool13, '14', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn14=addBatchNorm(net,conv14, '14', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn14, '14',slope=0.1)
    
    conv15=addConvolution(net, bn14, '15', ksize=1, noutput=512, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn15=addBatchNorm(net,conv15, '15', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn15, '15',slope=0.1)
    
    conv16=addConvolution(net, bn15, '16', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn16=addBatchNorm(net,conv16, '16', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn16, '16',slope=0.1)
    
    conv17=addConvolution(net, bn16, '17', ksize=1, noutput=512, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn17=addBatchNorm(net,conv17, '17', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn17, '17',slope=0.1)
    
    conv18=addConvolution(net, bn17, '18', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn18=addBatchNorm(net,conv18, '18', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn18, '18',slope=0.1)
     
    # added for detection
    conv19=addConvolution(net, bn18 , '19_a', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn19=addBatchNorm(net,conv19, '19_a', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn19, '19_a',slope=0.1)
    
    conv20=addConvolution(net, bn19 , '20_a', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn20=addBatchNorm(net,conv20, '20_a', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn20, '20_a',slope=0.1)
    
    conv21=addConvolution(net, bn20 , '21_a', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn21=addBatchNorm(net,conv21, '21_a', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn21, '21_a',slope=0.1)
    
    combined=addPassThrough(net,[bn19,bn21],'21','conv21')
    
    conv22=addConvolution(net, combined , '22_a', ksize=1, noutput=final_output, pad=0, stride=1, lr_mults=(1,2), decays=(1,0), hasReLU=False,hasBias=False)
    bn22=addBatchNorm(net,conv22, '22_a', hasBias=True, isTrain=True, lr_mult=(1,1), decay_mult=(1,0))
    addReLU(net, bn22, '22_a')
    
    addYoloV1Loss(net, 'train-loss', [bn22,tag], grid, class_num, bbox, threshold=0.1, noObjScale=0.5, coordScale=5., stage='TRAIN')
    addYoloV1Loss(net, 'val-loss', [bn22,tag], grid, class_num, bbox, threshold=0.1, noObjScale=0.5, coordScale=5., stage='TEST')
    
    exportModel(net, fname)
    

def get_DarkNet19_deploy(fname,class_num,bbox=2):
    final_output=5*bbox+class_num
    input_str='input: \"%s\"\ninput_shape{\n\tdim: 1\n\tdim: %d\n\tdim:%d\n\tdim:%d\n}\n'%('data',3,224,224)
    net=caffe.NetSpec()
    net.data=L.Layer()
    conv1=addConvolution(net, net.data, '1', ksize=3, noutput=32, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn1=addBatchNorm(net,conv1, '1', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn1, '1',slope=0.1)
    pool1=addPooling(net, bn1, '1', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv2=addConvolution(net, pool1, '2', ksize=3, noutput=64, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn2=addBatchNorm(net,conv2, '2', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn2, '2',slope=0.1)
    pool2=addPooling(net, bn2, '2', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv3=addConvolution(net, pool2, '3', ksize=3, noutput=128, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn3=addBatchNorm(net,conv3, '3', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn3, '3',slope=0.1)
    
    conv4=addConvolution(net, bn3, '4', ksize=1, noutput=64, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn4=addBatchNorm(net,conv4, '4', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn4, '4',slope=0.1)
    
    conv5=addConvolution(net, bn4, '5', ksize=3, noutput=128, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn5=addBatchNorm(net,conv5, '5', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn5, '5',slope=0.1)
    pool5=addPooling(net, bn5, '5', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv6=addConvolution(net, pool5, '6', ksize=3, noutput=256, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn6=addBatchNorm(net,conv6, '6', hasBias=True, isTrain=False,lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn6, '6',slope=0.1)
    
    conv7=addConvolution(net, bn6, '7', ksize=1, noutput=128, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn7=addBatchNorm(net,conv7, '7', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn7, '7',slope=0.1)
    
    conv8=addConvolution(net, bn7, '8', ksize=3, noutput=256, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn8=addBatchNorm(net,conv8, '8', hasBias=True, isTrain=False,lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn8, '8',slope=0.1)
    pool8=addPooling(net, bn8, '8', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv9=addConvolution(net, pool8, '9', ksize=3, noutput=512, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn9=addBatchNorm(net,conv9, '9', hasBias=True, isTrain=False,lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn9, '9',slope=0.1)
    
    conv10=addConvolution(net, bn9, '10', ksize=1, noutput=256, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn10=addBatchNorm(net,conv10, '10', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn10, '10',slope=0.1)
    
    conv11=addConvolution(net, bn10, '11', ksize=3, noutput=512, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn11=addBatchNorm(net,conv11, '11', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn11, '11',slope=0.1)
    
    conv12=addConvolution(net, bn11, '12', ksize=1, noutput=256, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn12=addBatchNorm(net,conv12, '12', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn12, '12',slope=0.1)
    
    conv13=addConvolution(net, bn12, '13', ksize=3, noutput=512, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn13=addBatchNorm(net,conv13, '13', hasBias=True, isTrain=False,lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn13, '13',slope=0.1)
    pool13=addPooling(net, bn13, '13', ksize=2, pad=0, stride=2, pool='MAX')
    
    conv14=addConvolution(net, pool13, '14', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn14=addBatchNorm(net,conv14, '14', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn14, '14',slope=0.1)
    
    conv15=addConvolution(net, bn14, '15', ksize=1, noutput=512, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn15=addBatchNorm(net,conv15, '15', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn15, '15',slope=0.1)
    
    conv16=addConvolution(net, bn15, '16', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn16=addBatchNorm(net,conv16, '16', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn16, '16',slope=0.1)
    
    conv17=addConvolution(net, bn16, '17', ksize=1, noutput=512, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn17=addBatchNorm(net,conv17, '17', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn17, '17',slope=0.1)
    
    conv18=addConvolution(net, bn17, '18', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn18=addBatchNorm(net,conv18, '18', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn18, '18',slope=0.1)
     
    # added for detection
    conv19=addConvolution(net, bn18 , '19_a', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn19=addBatchNorm(net,conv19, '19_a', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn19, '19_a',slope=0.1)
    
    conv20=addConvolution(net, bn19 , '20_a', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn20=addBatchNorm(net,conv20, '20_a', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn20, '20_a',slope=0.1)
    
    conv21=addConvolution(net, bn20 , '21_a', ksize=3, noutput=1024, pad=1, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn21=addBatchNorm(net,conv21, '21_a', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn21, '21_a',slope=0.1)
    
    combined=addPassThrough(net,[bn19,bn21],'21','conv21')
    
    conv22=addConvolution(net, combined , '22_a', ksize=1, noutput=final_output, pad=0, stride=1, lr_mults=(0,0), decays=(0,0), hasReLU=False,hasBias=False)
    bn22=addBatchNorm(net,conv22, '22_a', hasBias=True, isTrain=False, lr_mult=(0,0), decay_mult=(0,0))
    addReLU(net, bn22, '22_a')
    
    exportModel(net, fname, input_str,True)
    
def exportYoLoDarknet(dirname,train_img_path,train_annot,train_batch,val_img_path,val_annot,val_batch,class_num,bbox=2):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    get_DarkNet19_deploy(os.path.join(dirname,'deploy.prototxt'), class_num, bbox)
    get_YoLoV1_DarkNet19_trainval(os.path.join(dirname,'trainval.prototxt'), train_img_path, train_annot, val_img_path, val_annot, class_num, train_batch, val_batch, 7, bbox, datasize=0)
    
def get_extractionTrainval():
    pass
    
if len(sys.argv)>6:
    exportYoLoDarknet(sys.argv[1], sys.argv[6], '',int(sys.argv[2]), '', '' , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
elif len(sys.argv)>7:
    exportYoLoDarknet(sys.argv[1], sys.argv[6], sys.argv[7],int(sys.argv[2]), '', '' , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
elif len(sys.argv)>8:
    exportYoLoDarknet(sys.argv[1], sys.argv[6], sys.argv[7],int(sys.argv[2]), sys.argv[8], '' , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
elif len(sys.argv)>9:
    exportYoLoDarknet(sys.argv[1], sys.argv[6], sys.argv[7],int(sys.argv[2]), sys.argv[8], sys.argv[9] , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
else:
    exportYoLoDarknet(sys.argv[1], '', '',int(sys.argv[2]), '', '' , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    