
from abc import ABC, abstractmethod


class AbstractReport(ABC):
    dataset_tag = ''

    @abstractmethod
    def report(self, **kwargs):
        raise (NotImplementedError('Do not use abstract class method'))
