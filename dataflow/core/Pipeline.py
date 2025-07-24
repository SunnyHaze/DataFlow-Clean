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
        for k, v in vars(self).items():
            if isinstance(v, OperatorABC):
                setattr(self, k, AutoOP(v, self))
        self.forward()
        self.is_compiled = True
        
    def run(self):
        if self.is_compiled:
            for op_runtime in self.op_runtimes:
                op_runtime.func(**op_runtime.kwargs)
        else:
            self.forward()