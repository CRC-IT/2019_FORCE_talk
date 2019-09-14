
import random
from abc import abstractmethod, ABC
from collections import defaultdict


class DataGenerator(ABC):
    seed = random.randint(0, 2000000)
    children = defaultdict(list)
    initial_state = None
    current_state = None

    def __init__(self):
        self.seed = random.randint(0, 2000000)
        self.initial_state = random.getstate()
        self.current_state = random.getstate()
        self.children = defaultdict(list)

    @abstractmethod
    def return_synthetic_data(self):
        pass

    def reset(self):
        self.current_state = random.setstate(self.initial_state)
