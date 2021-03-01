# Face Recognition

We will be trying different models for performing face recognition.

### Model trained with omniglot dataset

Ran the original source code with normal SGD instead of the modified one, where
the final mean global accuracy was around 0.6475. The model had around 0.125 accuracy
for face recognition tasks.

Retraining that model on the face dataset helped achieve 0.4 mean evalulation accuracy in the one shot face recognition task.

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

#### RGB input instead of grayscale input

Instead of having a grayscale single channel input modified model and loader to take
rgb images.
Also modified laoder to take data from hdf5 file instead which improved image read time
significantly. Preoloading a large batches allows to utilize the GPU shortly more but doesnt really
save a lot of time as larger the preload longer the wait.

Trained a model where best validation accuracy was 0.3 evaluation accracy was 0.325


## Links

- Paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Code taken and modified further from: https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning
- Small section of data collected from: https://drive.google.com/drive/folders/0B5G8pYUQMNZnLTBVaENWUWdzR0E
