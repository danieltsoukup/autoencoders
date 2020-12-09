import unittest
from randnet import BatchAdaptiveDataGenerator, RandAE
import numpy as np


class TestBatchAdaptiveDataGenerator(unittest.TestCase):
    def setUp(self):
        self.data = np.random.randn(10000, 10)
        self.batch_size = 32
        self.epochs = 10
        self.generator = BatchAdaptiveDataGenerator(
            self.data,
            self.batch_size,
            self.epochs,
            subsample=0.5,
            verbose=True,
        )
        self.model = RandAE(input_dim=10, hidden_dims=[16, 8, 16])
        self.model.compile(optimizer="adam", loss="mse", run_eagerly=True)

    def test_fit(self):
        self.model.fit_generator(generator=self.generator, epochs=10, workers=1)
