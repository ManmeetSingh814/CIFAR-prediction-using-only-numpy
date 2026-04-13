import numpy as np

#image is [32,32,3] where filter is [32,3,3,3]
def Maxpooling(filmaps,poolingnumber=2):
    maxx = filmaps.shape[1]
    maxy = filmaps.shape[2]
    pooledmaps = np.zeros((filmaps.shape[0],maxx//2,maxy//2))
    for i in range(0,maxx,poolingnumber):
        for j in range(0,maxy,poolingnumber):
            smallfillmaps = filmaps[:,i:i+2,j:j+2]
            maxsfm = np.max(smallfillmaps,axis=(1,2))
            pooledmaps[:,i//2,j//2] = maxsfm
    return pooledmaps

def convolution(image, filters,pad=1,stride=1):
    image = np.pad(image,pad_width=[(pad,pad),(pad,pad),(0,0)],mode='constant')
    mapsx = (image.shape[0] - filters.shape[1])//stride + 1
    mapsy = (image.shape[1] - filters.shape[2])//stride + 1 
    filtermaps = np.zeros((filters.shape[0],mapsx,mapsy))
    for i in range(mapsy*mapsx):
        x = i%mapsx
        y = (i//mapsy)
        smallimage = image[x:x+3,y:y+3,:]
        for j in range(filters.shape[0]):
            f = filters[j]
            filtermaps[j,x,y] = np.sum(f*smallimage)
    return Maxpooling(filtermaps)



image = np.random.randn(32,32,3)
fillayer1 = np.random.randn(32,3,3,3)
fillayer2 = np.random.randn(64,3,3,32)

afterlayer1 = convolution(image,fillayer1)
afterlayer1 = np.transpose(afterlayer1,axes=(1,2,0))
afterlayer2 = convolution(afterlayer1,fillayer2)

flattenafterconvo = afterlayer2.flatten()
print(flattenafterconvo.shape)



# we need this because the convo function takes in the image parameter as (H,W,Depth) but it returns (number of layer,height,width) so transpose need before we can send it back to convo


#now we build the MLP from scratch

w1 = np.random.randn(flattenafterconvo.shape[0],1024)
w2 = np.random.randn()