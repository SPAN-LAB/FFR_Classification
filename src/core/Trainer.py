from abc import ABC, abstractmethod
from EEGSubject import EEGSubjectInterface, EEGSubject

from torch import nn

class TrainerInterface(ABC):
    model: nn.Module

    @abstractmethod
    def train(self, subject: EEGSubjectInterface, model_name: str, 
              num_epochs: int, lr: float, stopping_criteria: bool): ... 

    @abstractmethod
    def run(self, subject: EEGSubjectInterface): ... 

class Trainer(TrainerInterface):
    def __init__(self):
        self.model = None
        

    def train(self, subject: EEGSubject):
        return 1

    def run(self, subject: EEGSubject):
        return 1
