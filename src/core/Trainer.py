from abc import ABC, abstractmethod
from .EEGSubject import EEGSubjectInterface

from torch import nn

class TrainerInterface(ABC):
    model: nn.Module

    @abstractmethod
    def set_device(self, device_preference: str): ...

    @abstractmethod
    def train(self, subject: EEGSubjectInterface): ... 

    @abstractmethod
    def run(self, subject: EEGSubjectInterface): ... 

# class Trainer(TrainerInterface):