from abc import ABC, abstractmethod
from .Operator import OperatorABC
from dataflow.wrapper.auto_op import AutoOP

class PipelineABC(ABC):
    
    def __init__(self):
        self.op_runtimes = []
        self.is_compiled = False
    
    @abstractmethod
    def forward(self):
        """
        Main Function to run the pipeline
        """
        pass
    
    def compile(self):
        # TODO construct graph of keys and count objects
        for k, v in vars(self).items():
            if isinstance(v, OperatorABC):
                setattr(self, k, AutoOP(v, self))
        self.forward()
        self.forward = self._compiled_forward

    def _compiled_forward(self):
        # TODO add logic for Garbage Collection of Servings
        for op_runtime in self.op_runtimes:
            op_runtime.func(**op_runtime.kwargs)
