import os
import random
import numpy as np
import math
from PIL import Image

from ImageAugmentor import ImageAugmentor

# OmniglotLoader Class taken and modified from:
#
class FaceLoader:
    """

    This Class was constructed to read the faces, separate the
    training, validation and evaluation test. It also provides function for
    geting one-shot task batches.

    Attributes:
        dataset_path: path of Omniglot Dataset
        train_dictionary: dictionary of the files of the train set.
            This dictionary is used to load the batch for training and validation.
        evaluation_dictionary: dictionary of the evaluation set.
        image_width: self explanatory
        image_height: self explanatory
        batch_size: size of the batch to be used in training
        use_augmentation: boolean that allows us to select if data augmentation is
            used or not
        image_augmentor: instance of class ImageAugmentor that augments the images
            with the affine transformations referred in the paper

    """

    def __init__(self, dataset_path, use_augmentation, batch_size):
        """Inits FaceLoader with the provided values for the attributes.

        It also creates an Image Augmentor object and loads the train set and
        evaluation set into dictionaries for future batch loading.

        Arguments:
            dataset_path: path of Face dataset
            use_augmentation: boolean that allows us to select if data augmentation
                is used or not
            batch_size: size of the batch to be used in training
        """

        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.evaluation_dictionary = {}
        self.image_width = 105
        self.image_height = 105
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.example_each_person = 50
        self._train_male_faces = []
        self._train_female_faces = []

        self._validation_male_faces = []
        self._validation_female_faces = []

        self._evaluation_male_faces = []
        self._evaluation_female_faces = []

        self._current_train_face_index = 0
        self._current_validation_face_index = 0
        self._current_evaluation_face_index = 0

        self.load_dataset()

        if (self.use_augmentation):
            self.image_augmentor = self.createAugmentor()

    def load_dataset(self):
        """Loads the people into dictionaries

        Loads the face dataset and stores the available images for each
        person for each of the train and evaluation set.

        """

        train_path = os.path.join(self.dataset_path, 'Train')
        evaluation_path = os.path.join(self.dataset_path, 'Test')

        # First let's take care of the train alphabets
        for gender in os.listdir(train_path):
            gender_path = os.path.join(train_path, gender)

            current_gender_dictionary = {}

            for person in os.listdir(gender_path):
                person_path = os.path.join(gender_path, person)

                current_gender_dictionary[person] = os.listdir(person_path)

            self.train_dictionary[gender] = current_gender_dictionary

        # Now it's time for the evaluation faces
        for gender in os.listdir(evaluation_path):
            gender_path = os.path.join(evaluation_path, gender)

            current_gender_dictionary = {}

            for person in os.listdir(gender_path):
                person_path = os.path.join(gender_path, person)

                current_gender_dictionary[person] = os.listdir(person_path)

            self.evaluation_dictionary[gender] = current_gender_dictionary

    def createAugmentor(self):
        """ Creates ImageAugmentor object with the parameters for image augmentation

        Rotation range was set in -15 to 15 degrees
        Shear Range was set in between -0.3 and 0.3 radians
        Zoom range between 0.8 and 2
        Shift range was set in +/- 5 pixels

        Returns:
            ImageAugmentor object

        """
        rotation_range = [-15, 15]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 2]
        shift_range = [5, 5]

        return ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)

    def split_train_dataset(self):
        """ Splits the train set in train and validation

        Divide the train and validation with
        # a 80% - 20% split (8 vs 2 person)

        """

        male_people = list(self.train_dictionary['Male'].keys())
        female_people = list(self.train_dictionary['Female'].keys())
        number_of_males = len(male_people)
        number_of_females = len(female_people)

        male_indexes = random.sample(range(0, number_of_males - 1), int(0.8 * number_of_males))
        female_indexes = random.sample(range(0, number_of_females - 1), int(0.8 * number_of_females))
        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change
        male_indexes.sort(reverse=True)
        female_indexes.sort(reverse=True)

        for index in male_indexes:
            self._train_male_faces.append(male_people[index])
            male_people.pop(index)

        for index in female_indexes:
            self._train_female_faces.append(female_people[index])
            female_people.pop(index)

        # The remaining alphabets are saved for validation
        self._validation_male_faces = male_people
        self._validation_female_female_people = female_people

        self._evaluation_male_faces = list(self.evaluation_dictionary['Male'].keys())
        self._evaluation_female_faces = list(self.evaluation_dictionary['Female'].keys())

    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
        """ Loads the images and its correspondent labels from the path

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            path_list: list of images to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros(
            (number_of_pairs, self.image_height, self.image_width, 1)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair * 2]).convert('L')
            image = image.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()

            pairs_of_images[0][pair, :, :, 0] = image
            image = Image.open(path_list[pair * 2 + 1]).convert('L')
            image = image.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()

            pairs_of_images[1][pair, :, :, 0] = image
            if not is_one_shot_task:
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1
            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot_task:
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            pairs_of_images[0][:, :, :,
                               :] = pairs_of_images[0][random_permutation, :, :, :]
            pairs_of_images[1][:, :, :,
                               :] = pairs_of_images[1][random_permutation, :, :, :]

        return pairs_of_images, labels

    def get_train_batch(self):
        """ Loads and returns a batch of train images

        Each batch will iterate over the people list randomly picked
        and match it with itself as same image, and take another random pick
        from a different person.



        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """

        available_people = list(self._train_male_faces+self._train_female_faces)
        number_of_people= len(available_people)

        batch_images_path = []

        selected_person_indexes = [random.randint(0, number_of_people-1) for i in range(int(self.batch_size/2))]

        for index in selected_person_indexes:
            current_people = available_people[index]
            if current_people in self._train_male_faces:
                current_gender = "Male"
            else:
                current_gender = "Female"

            available_images = (self.train_dictionary[current_gender][current_people])
            image_path = os.path.join(self.dataset_path, 'Train', current_gender, current_people)

            # Random select a 3 indexes of images from the same person
            image_indexes = random.sample(range(0, self.example_each_person), 3)
            image = os.path.join(image_path, available_images[image_indexes[0]])
            batch_images_path.append(image)
            image = os.path.join(image_path, available_images[image_indexes[1]])
            batch_images_path.append(image)

            # Now let's take care of the pair of images from different person
            image = os.path.join(image_path, available_images[image_indexes[2]])
            batch_images_path.append(image)
            different_people = available_people[:]
            different_people.pop(index)
            different_person_index = random.sample(range(0, number_of_people - 1), 1)

            different_person = different_people[different_person_index[0]]
            if different_person in self._train_male_faces:
                different_gender = "Male"
            else:
                different_gender = "Female"

            available_images = self.train_dictionary[different_gender][different_person]
            image_indexes = random.sample(range(0, self.example_each_person), 1)
            image_path = os.path.join(self.dataset_path, 'Train', different_gender, different_person)
            image = os.path.join(image_path, available_images[image_indexes[0]])
            batch_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path, is_one_shot_task=False)

        # Get random transforms if augmentation is on
        if self.use_augmentation:
            images = self.image_augmentor.get_random_transform(images)

        return images, labels

    def get_one_shot_batch(self, support_set_size, is_validation):
        """ Loads and returns a batch for one-shot task images

        Gets a one-shot batch for evaluation or validation set, it consists in a
        single image that will be compared with a support set of images. It returns
        the pair of images to be compared by the model and it's labels (the first
        pair is always 1) and the remaining ones are 0's

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            male_faces = self._validation_male_faces
            female_faces = self._validation_female_faces
            image_folder_name = 'Train'
            dictionary = self.train_dictionary
        else:
            male_faces = self._evaluation_male_faces
            female_faces = self._evaluation_female_faces
            image_folder_name = 'Test'
            dictionary = self.evaluation_dictionary

        available_people = male_faces + female_faces
        number_of_people = len(available_people)

        batch_images_path = []

        test_person_index = random.sample(range(0, number_of_people), 1)

        # Get test image
        current_person = available_people[test_person_index[0]]
        if current_person in male_faces:
            current_gender = 'Male'
        else:
            current_gender = 'Female'
        available_images = dictionary[current_gender][current_person]

        image_indexes = random.sample(range(0, self.example_each_person), 2)
        image_path = os.path.join(self.dataset_path, image_folder_name, current_gender, current_person)

        test_image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(test_image)
        image = os.path.join(image_path, available_images[image_indexes[1]])
        batch_images_path.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_people = number_of_people
        else:
            number_of_support_people = support_set_size

        different_people = available_people[:]
        different_people.pop(test_person_index[0])

        if number_of_people < number_of_support_people:
            number_of_support_people = number_of_people

        support_people_indexes = random.sample(range(0, number_of_people - 1), number_of_support_people - 1)

        for index in support_people_indexes:
            current_person = different_people[index]
            if current_person in male_faces:
                current_gender = "Male"
            else:
                current_gender= "Female"
            available_images = dictionary[current_gender][current_person]
            image_path = os.path.join(self.dataset_path, image_folder_name, current_gender, current_person)

            image_indexes = random.sample(range(0, self.example_each_person), 1)
            image = os.path.join(image_path, available_images[image_indexes[0]])
            batch_images_path.append(test_image)
            batch_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path, is_one_shot_task=True)

        return images, labels

    def one_shot_test(self, model, support_set_size, number_of_tries,
                      is_validation):
        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """


        mean_global_accuracy = 0

        for _ in range(number_of_tries):
            images, _ = self.get_one_shot_batch(support_set_size, is_validation=is_validation)
            probabilities = model.predict_on_batch(images)

            # Added this condition because noticed that sometimes the outputs
            # of the classifier was almost the same in all images, meaning that
            # the argmax would be always by definition 0.
            if np.argmax(probabilities) == 0 and probabilities.std()>0.01:
                accuracy = 1.0
            else:
                accuracy = 0.0

            mean_global_accuracy += accuracy

        mean_global_accuracy /= number_of_tries

        print('\nMean global accuracy: ' + str(mean_global_accuracy))

        return mean_global_accuracy
