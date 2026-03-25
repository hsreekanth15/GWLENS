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
    def Fermat_Potential(self,x1,x2):
        pass

    @abstractmethod
    def Time_Delay(self,x1,x2,y1,y2):
        pass

    @abstractmethod
    def deflection(self,x):
        pass

    @abstractmethod
    def Hessian(self,x1,x2):
        pass

    @abstractmethod
    def Dimensionless_Frequency(self):
        pass

    @abstractmethod
    def solve_images(self, y1, y2, method="analytic"):
        pass


    def dimensionless_time_delay(self, y1, y2, method="analytic"):

        images = self.solve_images(y1, y2, method=method)

        for img in images:
            x1, x2 = img["position"]
            phi = self.Fermat_Potential(x1, x2, y1, y2)
            img["fermat_potential"] = phi

        phi_min = min(img["fermat_potential"] for img in images)

        for img in images:
            img["time_delay"] = img["fermat_potential"] - phi_min

        return images