# Face Recognition

We will be trying different models for performing face recognition.

## Siamese Neural Network for One Shote Learning

Attempting to build a model to perform face recognition.
Taking Siamese Network for One Shot Image Verification paper's approach
and applying it with a small face dataset.

In the first attempt the best validation accuracy lies around 0.225 and evaluation
accuracy is 0.125 for the one shot tasks.

The dataset is small and FaceLoader is not very effective at the moment making
the training very slow and not very accurate.

#### Face Loader Preloading 100 batches

Modified Face Loader to preload every 100 batches, where to create the full dataset
for the 200 batch it randomly picks 100 persons (in our case it will repeat 8 different person randomly)

Afterwards for each person it form pairs for one batch of data, half of it randomly paired with an image of itself
and another half randomly picking from a different class.

At the end the full preloaded set is shuffled so not necessarily batch at the end contains the same split.

The model was trained like before which led to higher validation accruacy (0.4) but seemed to decrease the
evalution accuracy to 0.1




## Links

- Paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Code taken and modified further from: https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning
- Small section of data collected from: https://drive.google.com/drive/folders/0B5G8pYUQMNZnLTBVaENWUWdzR0E
