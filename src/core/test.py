from EEGSubject import EEGSubject
from Plots import *
from Trainer import Trainer

something = EEGSubject \
                .init_from_filepath("../data/4T1015.mat") \
                .trim_by_index(0, 100) \
                .subaverage(5) \
                .fold(5)

trainer = Trainer()
trainer.train(subject=something) # 
participant = EEGSubject.init_from_filepath("../data/4T1015.mat")
trainer.run(participant) 

participant.trials[0].prediction