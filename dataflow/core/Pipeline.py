from abc import ABC, abstractmethod
from .Operator import OperatorABC
from dataflow.wrapper.auto_op import AutoOp

class PipelineABC(ABC):
    
    @abstractmethod
    def forward(self):
        """
        Main Function to run the pipeline
        """
        pass
    
    def autoop_register(self):
        for k, v in vars(self).items():
            if isinstance(v, OperatorABC):
                setattr(self, k, AutoOp(v))