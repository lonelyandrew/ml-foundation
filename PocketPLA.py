#!/usr/bin/env python3

import numpy as np


def load_dataset(path, verbose=False):
    '''Load the dataset from the path.

    Args:
        path (str): The path of the loading dataset.
        verbose (bool): Whether to display loading info.

    Returns:
        Return a list dataset whose element is a dict {x: [...], y: ...}.
    '''
    with open(path) as f:
        dataset = []
        for line in f:
            line = line.split()
            x = np.array([1] + line[:-1], dtype=float)
            y = int(line[-1])
            dataset.append({'x': x, 'y': y})
        if verbose:
            print(f'LOAD {len(dataset)} samples from {path}.')
        return dataset


class Perceptron:
    '''A Perceptron implementation.
    '''

    def __init__(self):
        '''Init the perceptron.
        '''
        self.weights = np.zeros(5)

    def __call__(self, x):
        '''Calculate the perceptron predict.

        Args:
            x (np.array): The input data.

        Returns:
            Return a binary label -1 or +1.
        '''
        return 1 if self.weights @ x > 0 else -1


class PerceptronLearningBase:
    '''The base class of PLAs.
    '''

    def __init__(self, dataset):
        '''Init the algorithm.

        Args:
            dataset (list): The training dataset.
        '''
        self.dataset = dataset
        self.perceptron = Perceptron()
        self.learning_rate = 1.0

    def update(self, sample):
        '''Update the perceptron.

        Args:
            sample (dict): An incorrectly-classified sample.
        '''
        delta = sample['x'] * sample['y'] * self.learning_rate
        self.perceptron.weights += delta

    def learn(self, learning_rate=1.0):
        '''The main process of learning.

        Args:
            learning_rate (int, optional): The learning rate.

        Returns:
            Return the update count.
        '''
        raise NotImplementedError


class NaiveCycleLearning(PerceptronLearningBase):
    '''Update perceptron with naive cycle order.
    '''

    def learn(self):
        '''Performe the learning process until no mistake detected.

        Returns:
            Return the update count.
        '''
        update_count = 0
        while True:
            updated = False
            for sample in self.dataset:
                if self.perceptron(sample['x']) != sample['y']:
                    updated = True
                    update_count += 1
                    self.update(sample)
            if not updated:
                break
        return update_count


class PocketLearning(PerceptronLearningBase):
    '''A Pocket Perceptron Learning (PLA) implementation.
    '''

    def __init__(self, dataset):
        '''Init the learner.

        Args:
            dataset (list): The training dataset.
        '''
        super().__init__(dataset)
        self.pocket_perceptron = Perceptron()
        self.pocket_error_rate = self.verify(self.pocket_perceptron, dataset)

    def learn(self, update=50):
        '''Performe the learning process until no mistake detected.

        Returns:
            Return the update count.
        '''
        for _ in range(update):
            while True:
                sample = np.random.choice(self.dataset)
                if (self.perceptron(sample['x']) != sample['y']):
                    self.update(sample)
                    error_rate = self.verify(self.perceptron, self.dataset)
                    if error_rate < self.pocket_error_rate:
                        self.pocket_perceptron.weights = np.copy(
                                self.perceptron.weights)
                        self.pocket_error_rate = error_rate
                    break

    def verify(self, perceptron, dataset):
        '''Verify the perceptron.

        Args:
            perceptron (Perceptron): The verifying perceptron.
            dataset (list): The verifying dataset.

        Returns:
            Return the error rate.
        '''
        error_count = 0
        for sample in dataset:
            if perceptron(sample['x']) != sample['y']:
                error_count += 1
        return error_count / len(dataset)


def question_15():
    dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_15_train.dat')
    learner = NaiveCycleLearning(dataset)
    update_count = learner.learn()
    for sample in dataset:
        assert learner.perceptron(sample['x']) == sample['y']
    print(f'UPDATE {update_count} TIMES')


def question_16():
    total_update_count = 0
    dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_15_train.dat')

    for _ in range(2000):
        shuffled_dataset = dataset[:]
        np.random.shuffle(shuffled_dataset)
        learner = NaiveCycleLearning(shuffled_dataset)
        update_count = learner.learn()
        for sample in dataset:
            assert learner.perceptron(sample['x']) == sample['y']
        total_update_count += update_count
    print(f'AVG UPDATE {total_update_count / 2000} TIMES')


def question_17():
    total_update_count = 0
    dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_15_train.dat')

    for _ in range(2000):
        shuffled_dataset = dataset[:]
        np.random.shuffle(shuffled_dataset)
        learner = NaiveCycleLearning(shuffled_dataset)
        learner.learning_rate = 0.5
        update_count = learner.learn()
        for sample in dataset:
            assert learner.perceptron(sample['x']) == sample['y']
        total_update_count += update_count
    print(f'AVG UPDATE {total_update_count / 2000} TIMES')


def question_18():
    train_dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_18_train.dat')
    test_dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_18_test.dat')

    total_error_rate = 0.0
    learner = PocketLearning(train_dataset)
    learner.learn()

    for i in range(2000):
        learner = PocketLearning(train_dataset)
        learner.learn()
        total_error_rate += learner.verify(learner.pocket_perceptron,
                                           test_dataset)
        print(f'ROUND #{i+1}: {total_error_rate/(i+1)}')


def question_19():
    train_dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_18_train.dat')
    test_dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_18_test.dat')

    total_error_rate = 0.0
    learner = PocketLearning(train_dataset)
    learner.learn()

    for i in range(2000):
        learner = PocketLearning(train_dataset)
        learner.learn()
        total_error_rate += learner.verify(learner.perceptron, test_dataset)
        print(f'ROUND #{i+1}: {total_error_rate/(i+1)}')


def question_20():
    train_dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_18_train.dat')
    test_dataset = load_dataset('/Users/shixiufeng/Desktop/hw1_18_test.dat')

    total_error_rate = 0.0
    learner = PocketLearning(train_dataset)
    learner.learn()

    for i in range(2000):
        learner = PocketLearning(train_dataset)
        learner.learn(update=100)
        total_error_rate += learner.verify(learner.pocket_perceptron,
                                           test_dataset)
        print(f'ROUND #{i+1}: {total_error_rate/(i+1)}')


if __name__ == '__main__':
    question_20()
