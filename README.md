## Fader_network
Our **Code** presents the Fader network that has for objective to generate different realistic versions
of an input image by changing the binary values of the attributes(-1/1) :
* 1 :
the image contains this attribute. 
* -1 : the image does not contain this attribute.
And to control some attributes of interest in the images, for which the transformations
are ill-defined and the learning is unsupervised, i.e. no image with the same content
but different attribute values is available.

We will reproduce the architecture described by the article. We will train it with the same
CelebA dataset as the article we will develop and compare our results with theirs. To start, we will
describe the method we will use with the choices we had to make. Then we will look at the results
and discuss the differences we have with the original results and their explanation of these differences.

 
##  Dataset:  
Get CelebA's aligned and cropped dataset at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. All images should be extracted and moved to the data/img_align_celeba/ folder. The number of images should be 202599. List_attr_celeba.txt contains the 40 attributes associated to each image in the dataset.
a suivre ...
