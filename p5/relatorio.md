A Survey on Transfer Learning
Sinno Jialin Pan and Qiang Yang Fellow, IEEE

transfer learning
machine learning algorithms make
predictions on the future data using statistical models that are
trained on previously collected labeled or unlabeled training
data [11], [12], [13]. 

Transfer learning allows the domains, tasks, and distributions used
in training and testing to be different. In the real world, we
observe many examples of transfer learning. For example,
we may find that learning to recognize apples might help to
recognize pears. Similarly, learning to play the electronic organ
may help facilitate learning the piano. The study of Transfer
learning is motivated by the fact that people can intelligently
apply knowledge learned previously to solve new problems
faster or with better solutions. 

The fundamental motivation
for Transfer learning in the field of machine learning was
discussed in a NIPS-95 workshop on “Learning to Learn”
1, which focused on the need for lifelong machine-learning
methods that retain and reuse previously learned knowledge.

Research on transfer learning has attracted more and
more attention since 1995

GoodFellow
In transfer learning, the learner must perform two or more different tasks, but we assume that many of the factors that explain the variations in P1 are relevant to the variations that need to be captured for learning P2.  This is typically understood in a supervised learning contexto, where the input is the same but the target may be of a different nature. For example, we may learn about one set of visual categories, such as cats and dogs, in the first setting, then learn about a different set of visual categories, such as ants and wasps, in the second setting. Many visual categories share low-level notons of edges and visual shapes, the effects of geometrocs changes, changes in lighting and domain adaptation can be achieved via representation learning. 



Dataset Augmentation
To make a machine learning model generalize better one need to train it on more data.  Of course, in practive, the ammount of data we have is limited.  One way to get around this problem is to create fake data and add it to the training set. 

This approach is not readily applicable to many tasks. For example, it is difficult to generate fake data ... but it has been particularly effective in object recognition. Operations like translating the training images a few pixels each direction  can improve generalization... rotating, scaling.

----
c
Image classification

Dubbed as one of the milestones in deep learning, this research paper “ImageNet Classification with Deep Convolutional Neural Networks” started it all. Even though deep learning had been around since the 70s with AI heavyweights Geoff Hinton, Yann LeCun and Yoshua Bengio working on Convolutional Neural Networks, AlexNet brought deep learning into the mainstream. Authored by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, this 2012 paper won the ImageNet Large Scale Visual Recognition Challenge with a 15.4% error rate. In fact, 2012 marked the first year when a CNN was used to achieve a top 5 test error rate of 15.4% and the next best research paper achieved an error rate of 26.2.  the paper was ground-breaking in its approach and brought the many concepts of deep learning into the mainstream. 

Goodfellow

The largest contest in object recognition is the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).  A dramatic moment in the meteoric rise of deep learning came when a convolutional network won this challenge for the first time and by a wide margin (Krizhevsky)

<<ILSVRC classification error rate>>


Resnext paper

Research on visual recognition is undergoing a transition
from “feature engineering” to “network engineering”
[25, 24, 44, 34, 36, 38, 14]. In contrast to traditional handdesigned
features (e.g., SIFT [29] and HOG [5]), features
learned by neural networks from large-scale data [33] require
minimal human involvement during training, and can
be transferred to a variety of recognition tasks [7, 10, 28].
Nevertheless, human effort has been shifted to designing
better network architectures for learning representations

Designing architectures becomes increasingly difficult
with the growing number of hyper-parameters (width2
, filter
sizes, strides, etc.), especially when there are many layers.

The VGG-nets [36] exhibit a simple yet effective strategy
of constructing very deep networks: stacking blocks of the same shape. This strategy is inherited
by ResNets [14] which stack modules of the same topology.
This simple rule reduces the free choices of hyperparameters,
and depth is exposed as an essential dimension
in neural networks. Moreover, we argue that the simplicity
of this rule may reduce the risk of over-adapting the hyperparameters
to a specific dataset. The robustness of VGGnets
and ResNets has been proven by various visual recognition
tasks [7, 10, 9, 28, 31, 14] and by non-visual tasks
involving speech [42, 30] and language [4, 41, 20].



ur neural networks, named ResNeXt (suggesting the
next dimension), outperform ResNet-101/152 [14], ResNet200
[15], Inception-v3 [39], and Inception-ResNet-v2 [37]
on the ImageNet classification dataset. In particular, a
101-layer ResNeXt is able to achieve better accuracy than
ResNet-200 [15] but has only 50% complexity. Moreover,
ResNeXt exhibits considerably simpler designs than all Inception
models. ResNeXt was the foundation of our submission
to the ILSVRC 2016 classification task, in which
we secured second place


MACHINE TYPE: GPU+ HOURLY
REGION: NY2
PRIVATE IP: 
10.64.63.8
COPIED
PUBLIC IP: 
184.105.217.44
COPIED
RAM: 30 GB
CPUS: 8
HD: 107.5 KB / 50 GB
GPU: 8 GB
NETWORK: PAPERSPACE

Cyclical Learning Rate

powerful technique to select a range of learning rates for a neural network
The trick is to train a network starting from a low learning rate and increase the learning rate exponentially for every batch.

Record the learning rate and training loss for every batch. Then, plot the loss and the learning rate. Typically, it looks like this:

