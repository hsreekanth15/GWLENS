#Base class for lens models in GWLENS

from abc import ABC, abstractmethod

class Lens(ABC):

    """
    Base class for Lens models.
    Abstract methods are defined here...

    Creating a subclass requires implementing the abstract methdos in base class
    
    
    """

    def __init__(self,**params):
        self.params = params

    @abstractmethod
    def potential(self,x):
        pass

    @abstractmethod
    def time_delay(self,x1,x2):
        pass

    @abstractmethod
    def deflection(self,x):
        pass

    @abstractmethod
    def Image_count(self):
        pass

    @abstractmethod
    def Hessian(self,x1,x2):
        pass

    @abstractmethod
    def Dimensionless_Frequency(self):
        pass