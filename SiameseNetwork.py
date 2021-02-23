import os

import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD

from FaceLoader import FaceLoader

class SiameseNetwork:


    def __init__(self, dataset_path,  learning_rate, batch_size, use_augmentation,
                 l2_param, tensorboard_log_path):

        self.input_shape = (105, 105, 1)
        self.model = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size # u want to have a image loader here that takes images as pair from # DEBUG:
        self.summary_writer = tf.summary.create_file_writer(tensorboard_log_path)
        self.face_loader = FaceLoader(dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size)
        self._construct_network(l2_param)

    def _construct_network(self, l2_param):

        encoder = Sequential()
        encoder.add(Conv2D(filters=64, kernel_size=(10, 10),
                           activation='relu', input_shape=self.input_shape,
                           kernel_regularizer=l2(l2_param['Conv1']),
                           name='Conv1'))
        encoder.add(MaxPool2D())

        encoder.add(Conv2D(filters=128, kernel_size=(7, 7),
                           activation='relu',
                           kernel_regularizer=l2(l2_param['Conv2']),
                           name='Conv2'))
        encoder.add(MaxPool2D())

        encoder.add(Conv2D(filters=128, kernel_size=(4, 4),
                           activation='relu',
                           kernel_regularizer=l2(l2_param['Conv3']),
                           name='Conv3'))
        encoder.add(MaxPool2D())

        encoder.add(Conv2D(filters=256, kernel_size=(4, 4),
                           activation='relu',
                           kernel_regularizer=l2(l2_param['Conv4']),
                           name='Conv4'))
        encoder.add(MaxPool2D())

        encoder.add(Flatten())
        encoder.add(Dense(units=4096, activation='sigmoid', kernel_regularizer=l2(l2_param['Dense1']), name='Dense1'))

        input_img_1 = Input(self.input_shape)
        input_img_2 = Input(self.input_shape)

        encoded_img_1 = encoder(input_img_1)
        encoded_img_2 = encoder(input_img_2)

        l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0]-tensors[1]))
        l1_distance = l1_distance_layer([encoded_img_1, encoded_img_2])

        prediction = Dense(units=1, activation='sigmoid')(l1_distance)

        self.model = Model(inputs=[input_img_1, input_img_2], outputs=prediction)

        optimizer = SGD(lr=self.learning_rate,momentum=0.5,name="SGD")

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=optimizer)

    def _write_logs_to_tensorboard(self, current_iteration, train_losses,
                                    train_accuracies, validation_accuracy,
                                    evaluate_each):
        """ Writes the logs to a tensorflow log file

        This allows us to see the loss curves and the metrics in tensorboard.
        If we wrote every iteration, the training process would be slow, so
        instead we write the logs every evaluate_each iteration.

        Arguments:
            current_iteration: iteration to be written in the log file
            train_losses: contains the train losses from the last evaluate_each
                iterations.
            train_accuracies: the same as train_losses but with the accuracies
                in the training set.
            validation_accuracy: accuracy in the current one-shot task in the
                validation set
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
        """

        # Write to log file the values from the last evaluate_every iterations

        with self.summary_writer.as_default():
            for index in range(0, evaluate_each):
                tf.summary.scalar("train_loss", train_losses[index], step=current_iteration - evaluate_each + index + 1)
                tf.summary.scalar("train_acc", train_accuracies[index], step=current_iteration - evaluate_each + index + 1)
                if index == (evaluate_each - 1):
                    tf.summary.scalar("one_shot_val_acc", validation_accuracy, step=current_iteration - evaluate_each + index + 1)

        self.summary_writer.flush()

    def train_network(self, number_of_iterations, support_set_size,
                      final_momentum, momentum_slope, evaluate_each,
                      model_name):

        self.face_loader.split_train_dataset()

        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0
        earrly_stop = 0
        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0

        for iteration in range(number_of_iterations):
            images, labels = self.face_loader.get_train_batch()
            train_loss, train_accuracy = self.model.train_on_batch(images, labels)


            # Decay learning rate 1 % per 500 iterations (in the paper the decay is
            # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
            if (iteration + 1) % 500 == 0:
                K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * 0.99)
            if K.get_value(self.model.optimizer.momentum) < final_momentum:
                K.set_value(self.model.optimizer.momentum, K.get_value(self.model.optimizer.momentum) + momentum_slope)

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy

            # validation set
            count += 1
            if ((iteration+1) % 50) == 0:
                print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(
                      self.model.optimizer.lr)))

            if (iteration + 1) % evaluate_each == 0:
                number_of_runs_per_person = 40
                # use a support set size equal to the number of character in the alphabet
                validation_accuracy = self.face_loader.one_shot_test(self.model, support_set_size, number_of_runs_per_person, is_validation=True)

                self._write_logs_to_tensorboard(iteration, train_losses, train_accuracies, validation_accuracy, evaluate_each)
                count = 0

                # Some hyperparameters lead to 100%, although the output is almost the same in
                # all images.
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' + str(best_validation_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # Save the model
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration

                        model_json = self.model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + '.json', "w") as json_file:
                            json_file.write(model_json)
                        self.model.save_weights('models/' + model_name + '.h5')

            # If accuracy does not improve for 10000 batches stop the training
            if iteration - best_accuracy_iteration > 10000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' + str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        print('Trained Ended!')
        return best_validation_accuracy
