import numpy as np

def shuffle_batch(image,min_class,count):
  data_sequence = np.random.permutation(min(count,image.shape[0]))
  return image[data_sequence[:min_class]]


def relabel(unsupervised_train, unsup_label,num_class,pixel_per_image):

  mean_class = unsup_label.shape[0]/num_class
  
  unsup_image={}
  for i in range(num_class):
    unsup_image[i] = np.zeros((mean_class,pixel_per_image),dtype=np.uint8)

  count = np.zeros((num_class),dtype=np.int32)
  
  for i in range(unsup_label.shape[0]):
    #print i
    temp_label = unsup_label[i]
    if temp_label >9:
      continue
    if count[temp_label] >= mean_class:
      count[temp_label] += 1
    else:  
      unsup_image[temp_label][count[temp_label]] = unsupervised_train[i]
      count[temp_label] += 1
  
  min_class = np.min(count)
  print 'number of min class example :',min_class
  
  for i in range(num_class):
    #unsup_image[i] = np.reshape(unsup_image[i][:min_class],(-1,1,pixel_per_image))
    unsup_image[i] = np.reshape(shuffle_batch(unsup_image[i],min_class,count[i]),(-1,1,pixel_per_image))
  
  new_unsup_image = unsup_image[0]
  for i in range(num_class-1):
    new_unsup_image=np.concatenate((new_unsup_image,unsup_image[i+1]),axis=1)
  
  
  new_unsup_label = np.tile(range(num_class),min_class)
  new_unsup_image = np.reshape(new_unsup_image,(-1,pixel_per_image))
  
  new_unsup_train = {'image':new_unsup_image, 'label':new_unsup_label}

  return new_unsup_train


