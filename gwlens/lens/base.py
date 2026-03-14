#Base class for lens models in GWLENS

from abc import ABC, abstractmethod

class Lens(ABC):

    def __init__(self,**params):
        self.params = params

    @abstractmethod
    def potential(self):
        pass

    @abstractmethod
    def time_delay(self):
        pass

    @abstractmethod
    def deflection(self):
        pass

    @abstractmethod
    def Image_count(self):
        pass

    @abstractmethod
    def Hessian(self):
        pass

    @abstractmethod
    def Dimensionless_Frequency(self):
        pass