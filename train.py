

from SiameseNetwork import SiameseNetwork

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def main():
    dataset_path = 'Face Dataset'
    use_augmentation = True
    learning_rate = 10e-4
    batch_size = 64

    # l2-regularization penalization for each layer
    l2_param = {}
    l2_param['Conv1'] = 1e-2
    l2_param['Conv2'] = 1e-2
    l2_param['Conv3'] = 1e-2
    l2_param['Conv4'] = 1e-2
    l2_param['Dense1'] = 1e-4

    # Path where the logs will be saved

    model_name = 'siamese_net_lr10e-4_v2'

    tensorboard_log_path = './logs/'+model_name
    siamese_network = SiameseNetwork(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, use_augmentation=use_augmentation,
        l2_param=l2_param,
        tensorboard_log_path=tensorboard_log_path
    )
    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = -1
    evaluate_each = 1000
    number_of_train_iterations = 1000000


    validation_accuracy = siamese_network.train_network(number_of_iterations=number_of_train_iterations,
                                                                support_set_size=support_set_size,
                                                                final_momentum=momentum,
                                                                momentum_slope=momentum_slope,
                                                                evaluate_each=evaluate_each,
                                                                model_name=model_name)
    if validation_accuracy == 0:
        evaluation_accuracy = 0
    else:
        # Load the weights with best validation accuracy
        siamese_network.model.load_weights('./models/'+ model_name +'.h5')
        evaluation_accuracy = siamese_network.face_loader.one_shot_test(siamese_network.model, 20, 40, False)

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))


if __name__ == "__main__":
    main()
