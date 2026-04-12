import numpy as np

#image is [32,32,3] where filter is [32,3,3,3]
def convolution(image, filters,pad=1,stride=1):
    image = np.pad(image,pad_width=[(pad,pad),(pad,pad),(0,0)],mode='constant')
    mapsx = (image.shape[0] - filters.shape[1])//stride + 1
    mapsy = (image.shape[1] - filters.shape[2])//stride + 1 
    filtermaps = np.zeros((filters.shape[0],mapsx,mapsy))
    for i in range(mapsy*mapsx):
        x = i%32
        y = (i//32)
        smallimage = image[x:x+3,y:y+3,:]
        for j in range(filters.shape[0]):
            f = filters[j]
            filtermaps[j,x,y] = np.sum(f*smallimage)
    pooledmaps = np.zeros((filters.shape[0],mapsx//2,mapsy//2))
    for i in range(0,mapsx,2):
        for j in range(0,mapsy,2):
            smallfiltermaps = filtermaps[:,i:i+2,j:j+2]
            maxd = np.max(smallfiltermaps,axis=(1,2))
            pooledmaps[range(filters.shape[0]),i//2,j//2]=maxd
    return pooledmaps.shape

        

image = np.random.randn(32,32,3)
filters = np.random.randn(32,3,3,3)
print(convolution(image,filters))