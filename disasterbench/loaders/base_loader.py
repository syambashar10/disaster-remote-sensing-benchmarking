from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    """
    Base class for all dataset loaders.

    Every dataset-specific loader must implement these methods.
    This keeps the framework consistent even when datasets have different raw formats.
    """

    @abstractmethod
    def list_samples(self):
        """
        Return a list of available dataset samples.
        """
        raise NotImplementedError

    @abstractmethod
    def load_sample(self, index):
        """
        Load one sample and return it in the common internal schema.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_pairing(self):
        """
        Check whether images, annotations, masks, and metadata match correctly.
        """
        raise NotImplementedError

    @abstractmethod
    def get_statistics(self):
        """
        Return dataset-level statistics.
        """
        raise NotImplementedError
