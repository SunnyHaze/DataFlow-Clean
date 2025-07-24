from typing import ParamSpec, TypeVar, Generic, Protocol, Any, Dict
from functools import wraps
import inspect
from dataflow.logger import get_logger
from dataflow.core import OperatorABC
from tqdm import tqdm
import pandas as pd
P = ParamSpec("P")
R = TypeVar("R")

class HasRun(Protocol[P, R]):
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        这里写一份通用的 run 文档也可以，
        不写的话会被下面动态拷贝原 operator.run 的 __doc__。
        """
        ...
        
class OPRuntime:
    def __init__(self, operator: OperatorABC, func: callable, args: Dict[str, Any]):
        self.op = operator
        self.func = func
        self.kwargs = args

    def __repr__(self):
        return f"OPRuntime(operator={repr(self.op)}, func={self.func.__qualname__}, args={self.kwargs})"        

class AutoOp(Generic[P, R]):
    """
    自动化运行 Operator 的 Wrapper。

    在静态检查/IDE 里，AutoOp.run 的签名和 operator.run 完全一致。
    运行时，会把 operator.run 的 __doc__ 和 __signature__ 也拷过来，
    这样 help(bw.run) 时能看到原 operator 的文档。
    """
    op_runtimes = []
    
    def __init__(self, operator: HasRun[P, R]):
        self._operator = operator
        self._logger = get_logger()
        
        # 动态拷贝 operator.run 的 __doc__ 和 inspect.signature
        self._orig_run = operator.run
        self._signature = inspect.signature(self._orig_run)
        self.__doc__ = self._orig_run.__doc__

    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        这里写一份通用的 run 文档也可以，
        不写的话会被上面动态拷贝原 operator.run 的 __doc__。
        """
        self._signature = inspect.signature(self._orig_run)
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        final_kwargs = bound_args.arguments  # OrderedDict

        # 添加一条运行记录
        self.__class__.op_runtimes.append(OPRuntime(self._operator, self._orig_run, dict(final_kwargs)))
    
    @classmethod
    def run_all(cls):
        for op_runtime in cls.op_runtimes:
            op_runtime.func(**op_runtime.kwargs)