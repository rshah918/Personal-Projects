import numpy as np
import mnist
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
class Conv3x3:
    #A convolution layer using 3x3 filters
    def __init__(self, num_filters, filter_size=3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        #filters is a 3d array with dimensions (num_filters, 3, 3)
        #Divided by 9 to reduce the variance of the initial values
        self.filters = np.random.randn(num_filters, filter_size,filter_size) / 9

    def iterate_regions(self, image):
        '''Generates all possible 3x3 image regions using valid padding.
            Image is a 2D numpy array'''
        h, w = image.shape

        for i in range(h-(self.filter_size-1)):
            for j in range(w-(self.filter_size-1)):
                im_region = image[i:(i + self.filter_size),j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        '''Performs a forward pass of the conv layer using the given input.
            Returns a 3D numpy array with dimensions (h,w,num_filters).
            Input is a 2D numpy array'''
        h, w = input.shape
        self.last_input = input
        output = np.zeros((h - (self.filter_size - 1), w - (self.filter_size - 1), self.num_filters))#output array is smaller after filter sweep

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1,2))
        '''each element contains an array with convolution results of each filter
                        for that pixel'''
        return output
    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        #each output pixel is equal to dot product of filter and image window
            #dout/dfilter = image window
            #so perform elementwise multiplication between image window and dl/dout window
            #elementwise subtract filter pixels from gradient, multiply by learning rate
        d_L_d_filter = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                #print(im_region.shape)

                d_L_d_filter[f] += d_L_d_out[i, j, f] * im_region

        self.filters = self.filters - (learn_rate * d_L_d_filter)
        return None

class MaxPool:
    def iterate_regions(self, image):
        '''generates a non-overlapping 2x2 image to pool over.
            Image is a 2x2 numpy array'''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]#shift window by 2
                yield im_region, i, j

    def forward(self, input):
        '''Performs a forward pass of the maxpool layer using the given input.
            Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
            input is a 3d numpy array with dimensions (h, w, num_filters)'''
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
          output[i, j] = np.amax(im_region, axis=(0, 1))
        return output
    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
          h, w, f = im_region.shape
          amax = np.amin(im_region, axis=(0, 1))

          for i2 in range(h):
            for j2 in range(w):
              for f2 in range(f):
                # If this pixel was the max value, copy the gradient to it.
                if im_region[i2, j2, f2] == amax[f2]:
                  d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
class Softmax:
  # A standard fully-connected layer with softmax activation.

    def __init__(self, nodes):
        self.biases = np.zeros(nodes)
        self.nodes = nodes
        self.weights_initialized = False

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        if self.weights_initialized == False:
            # We divide by input_len to reduce the variance of our initial values
            self.weights = np.random.randn(np.prod(input.shape), self.nodes) / np.prod(input.shape)
            self.weights_initialized = True
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        '''Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float'''

         # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
          if gradient == 0:
            continue

          # e^totals
          t_exp = np.exp(self.last_totals)

          # Sum of all e^totals
          S = np.sum(t_exp)

          # Gradients of out[i] against totals
          d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
          d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

          # Gradients of totals against weights/biases/input
          d_t_d_w = self.last_input
          d_t_d_b = 1
          d_t_d_inputs = self.weights

          # Gradients of loss against totals
          d_L_d_t = gradient * d_out_d_t

          # Gradients of loss against weights/biases/input
          d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
          d_L_d_b = d_L_d_t * d_t_d_b
          d_L_d_inputs = d_t_d_inputs @ d_L_d_t

          #update weights/biases
          self.weights -= learn_rate * d_L_d_w
          self.biases -= learn_rate * d_L_d_b

          return d_L_d_inputs.reshape(self.last_input_shape)
# We only use the first 1k testing examples (out of 10k total)
# in the interest of time. Feel free to change this if you want.
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool()                  # 26x26x8 -> 13x13x8
softmax = Softmax(10) # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # Normalize and resize
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)


  return loss, acc
'----------------------------------------------------------------------------------'
# get data
train_images = mnist.train_images()[:100]
train_labels = mnist.train_labels()[:100]
test_images = mnist.test_images()[:100]
test_labels = mnist.test_labels()[:100]
#hyperparameters
FILTER_SIZE=4
SOFTMAX_NODES = 10
NUM_FILTERS = 4
LEARNING_RATE = 0.03
NUM_EPOCHS = 5
#initialize layer objects
conv = Conv3x3(NUM_FILTERS, FILTER_SIZE)   # 28x28x1 -> 26x26x8
pool = MaxPool()                 # 26x26x8 -> 13x13x8
softmax = Softmax(SOFTMAX_NODES) # 13x13x8 -> 10
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print(''' _____                       _       _   _                   _   _   _                      _   _   _      _                      _
/  __ \                     | |     | | (_)                 | | | \ | |                    | | | \ | |    | |                    | |
| /  \/ ___  _ ____   _____ | |_   _| |_ _  ___  _ __   __ _| | |  \| | ___ _   _ _ __ __ _| | |  \| | ___| |___      _____  _ __| | __
| |    / _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| '_ \ / _` | | | . ` |/ _ \ | | | '__/ _` | | | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
| \__/\ (_) | | | \ V / (_) | | |_| | |_| | (_) | | | | (_| | | | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   <
 \____/\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|\__,_|_| \_| \_/\___|\__,_|_|  \__,_|_| \_| \_/\___|\__| \_/\_/ \___/|_|  |_|\_\

''')
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print("------------------------------------------------------------------------------------------------------------------------------------ ")
# Train the CNN for 3 epochs

print("""\n\t\tINITIALIZING NETWORK TRAINING\n""")


for epoch in range(NUM_EPOCHS):
  print('\t\t------ Training Epoch %d ------' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0

  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
      print(
        '\n \t [Step %d] Past 100 steps: Accuracy: %d%% \n' %
        (i + 1, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label, LEARNING_RATE)
    loss += l
    num_correct += acc
  LEARNING_RATE = LEARNING_RATE-0.002

# Test the CNN
print('\n\t\t------ Testing the CNN ------')
image_index = np.random.randint(0,100)

'''try:
    size = 26, 26
    outfile = "eight.jpg"
    img  = Image.open("eight.jpg")
    img = img.convert(mode="L")
    img.thumbnail(size, Image.ANTIALIAS)
    img.save(outfile, "JPEG")
    pix = np.array(img)
except IOError:
    print("Didnt work")

class_probabilities, l, acc = forward(pix, 8)

prediction = np.amax(class_probabilities)
print("\t\tTest Image Label: 8")
print("\t\tNeural Network Prediction: ", np.where(class_probabilities == prediction)[0][0])

plt.imshow(pix, cmap ='Greys')#display image
plt.show()'''

loss = 0
num_correct = 0
for im, label in zip(test_images[image_index-5: image_index], test_labels[image_index-5: image_index]):
  print("\t\t--------------------------------- \n")
  print("\t\tActual Image Label: ", label)
  class_probabilities, l, acc = forward(im, label)
  prediction = np.amax(class_probabilities)
  print("\t\tNeural Network Prediction: ", np.where(class_probabilities == prediction)[0][0])
  plt.imshow(im, cmap ='Greys')#display image
  plt.show()
  print("\t\t--------------------------------- ")

  loss += l
  num_correct += acc

num_tests = 5
print(num_tests)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', (num_correct / 5)*100, "%")
