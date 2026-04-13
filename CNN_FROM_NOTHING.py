import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets
cifar = datasets.CIFAR10(root='./cifar',train=True,download=True)
cifar_data = cifar.data/255 #shape is [50000,32,32,3]
cifar_labels = cifar.targets
print(cifar_data.shape)

#image is [32,32,3] where filter is [32,3,3,3]
def Maxpooling(filmaps,poolingnumber=2):
    maxx = filmaps.shape[2]
    maxy = filmaps.shape[3]
    pooledmaps = np.zeros((filmaps.shape[0],filmaps.shape[1],maxx//2,maxy//2))
    for i in range(0,maxx,poolingnumber):
        for j in range(0,maxy,poolingnumber):
            smallfillmaps = filmaps[:,:,i:i+2,j:j+2]
            maxsfm = np.max(smallfillmaps,axis=(2,3))
            pooledmaps[:,:,i//2,j//2] = maxsfm
    return pooledmaps

def convolution(image, filters,pad=1,stride=1):
    image = np.pad(image,pad_width=[(0,0),(pad,pad),(pad,pad),(0,0)],mode='constant')
    mapsx = (image.shape[1] - filters.shape[1])//stride + 1
    mapsy = (image.shape[2] - filters.shape[2])//stride + 1 
    filtermaps = np.zeros((image.shape[0],filters.shape[0],mapsx,mapsy))
    for i in range(mapsy*mapsx):
        x = i%mapsx
        y = (i//mapsy)
        smallimage = image[:,x:x+3,y:y+3,:]
        for j in range(filters.shape[0]):
            f = filters[j]
            filtermaps[:,j,x,y] = np.sum(f*smallimage,axis=(1,2,3)) #numpy comes in clutch with its smart indexing here
    return Maxpooling(filtermaps)



image = cifar_data[:10] /255
fillayer1 = np.random.randn(32,3,3,3)
fillayer2 = np.random.randn(64,3,3,32)

afterlayer1 = convolution(image,fillayer1)
afterlayer1 = np.transpose(afterlayer1,axes=(0,2,3,1))
afterlayer2 = convolution(afterlayer1,fillayer2)
afterlayer2=afterlayer2.reshape(10,-1)
print(afterlayer2.shape)



# we need this because the convo function takes in the image parameter as (H,W,Depth) but it returns (number of layer,height,width) so transpose need before we can send it back to convo


#now we build the MLP from scratch

w1 = np.random.randn(4096,1024) *0.01
w2 = np.random.randn(1024,128)*0.01
w3 = np.random.randn(128,10)*0.01

bias1 = np.random.randn(1024)*0.01
bias2 = np.random.randn(128)*0.01
bias3 = np.random.randn(10)*0.01

mw1 = np.zeros((4096,1024))
mw2 = np.zeros((1024,128))
mw3 = np.zeros((128,10))
vw1 = np.zeros((4096,1024))
vw2 = np.zeros((1024,128))
vw3 =np.zeros((128,10))


for epoch in range(0,25):
    losss =[]
    for i in range(0,1000,32):
        pilbatch = cifar_data[i:i+32]#32,32,32,3
        pilbatchcl1 = convolution(pilbatch,fillayer1)
        pilbatchcl1 = np.transpose(pilbatchcl1,axes=(0,2,3,1))
        pilbatchcl2 = convolution(pilbatchcl1,fillayer2)
        pilbatchcl2 = pilbatchcl2.reshape(32,-1)

        labelbatch = cifar_labels[i:i+32] # 32,1
        layer1 = np.dot(pilbatchcl2,w1) + bias1      #pilbatch is [32,784] w1 is [784,128] so layer1 = [32,128] numpy handles the multidimensionality
        postrelulayer1 = np.maximum(0,layer1)
        layer2 = np.dot(postrelulayer1,w2) + bias2
        postrelulayer2 = np.maximum(0,layer2)
        layer3 = np.dot(postrelulayer2,w3) + bias3
        softlayer3 = np.exp(layer3)    #crossEntropy
        softlayer3 = softlayer3/np.sum(softlayer3 , axis=1, keepdims=True)
        loss = -np.mean(np.log(softlayer3[range(32),labelbatch]))
        losss.append(loss)
        hot_one = np.zeros((32,10))
        hot_one[range(32),labelbatch] = 1
        dlossbydlayer3 = softlayer3 - hot_one
        dlossbydlayer3 = dlossbydlayer3/32 # a real pain in the ass to debug basically average them out for loss
        dlossbydweight3 = np.dot(np.transpose(postrelulayer2),dlossbydlayer3)#one down two to go
        dlossbydpostrelulayer2 = np.dot(dlossbydlayer3,np.transpose(w3))
        dlossbydlayer2 = dlossbydpostrelulayer2 * (layer2 > 0)
        dlossbydweight2 = np.dot(np.transpose(postrelulayer1),(dlossbydlayer2))
        dlossbydpostrelulayer1 = np.dot(dlossbydlayer2,np.transpose(w2))
        dlossbylayer1 = dlossbydpostrelulayer1 * (layer1 > 0)
        dlossbydweight1 = np.dot(np.transpose(pilbatchcl2),dlossbylayer1)

        learningrate = 0.01
        mw1=(0.9*mw1+dlossbydweight1*0.1)
        mw2=(0.9*mw2+dlossbydweight2*0.1)           #implementing Momentum and adaptive learning rate
        mw3=(0.9*mw3+dlossbydweight3*0.1)
        vw1 = (0.999*vw1+0.001*(dlossbydweight1 * dlossbydweight1))
        vw2 = (0.999*vw2+0.001*(dlossbydweight2 * dlossbydweight2))
        vw3 = (0.999*vw3+0.001*(dlossbydweight3 * dlossbydweight3))
        w1 = w1 - (learningrate * mw1)/(np.sqrt(vw1) + 0.0000000001)
        w2 = w2 - (learningrate * mw2)/(np.sqrt(vw2) + 0.0000000001)
        w3 = w3 - (learningrate * mw3)/(np.sqrt(vw3) + 0.0000000001)
        bias3 = bias3 - learningrate * np.sum(dlossbydlayer3, axis=0)
        bias2 = bias2 - learningrate * np.sum(dlossbydlayer2,axis=0)
        bias1 = bias1 - learningrate * np.sum(dlossbylayer1,axis=0)
        print("batch ",i)
    print(np.mean(losss))

