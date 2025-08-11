import copy
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataflow.core.Operator import OperatorABC
from dataflow.pipeline.nodes import OperatorNode, KeyNode
from dataflow.wrapper.auto_op import AutoOP, OPRuntime
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OPERATOR_CLASSES, LLM_SERVING_CLASSES

from dataflow.logger import get_logger

class PipelineABC(ABC):
    def __init__(self):
        # list of dict, contains `OPRuntime` class and parameters for `operator.run()`
        self.op_runtimes:list[OPRuntime] = [] 
        
        # accumulated keys in each operators, index 0 refers to the keys before the first operator
        self.accumulated_keys = []  # list of lists, each sublist contains keys before the operator
        
        # other items
        self.logger = get_logger()
        self.active_llm_serving = None
        # self.serving_resources = defaultdict(dict)
        # self.serving_reference_counter = Counter()
        
        self.op_nodes_list : list[OperatorNode] = []
        self.llm_serving_list = [] # list of LLMServing objects
        self.llm_serving_counter = Counter() # count of LLMServing objects
    
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
                setattr(self, k, AutoOP(v, k, self)) 
        self.forward()
        # after call forward, call back function in AutoOP will add the OPRuntime object to self.op_runtimes
        
        self.forward = self._compiled_forward
        self.logger.info(
            f"Compiling pipeline and validating key integrity "
            f"across {len(self.op_runtimes)} operator runtimes."
        )
        self._build_operator_nodes_graph()
        # self._build_serving_resources_map()
        
    def _build_operator_nodes_graph(self):
        """
        Build a graph of operator nodes, each node contains the operator object and its storage.
        """
        for op_runtime in self.op_runtimes:
            llm_serving_obj, storage_obj = None, None
            # get llm_serving object from the operator
            for _, v in vars(op_runtime.op).items():
                if isinstance(v, LLM_SERVING_CLASSES):
                    llm_serving_obj = v
            # get storage object from the function dict
            storage_obj = op_runtime.kwargs.pop("storage", None)
            
            assert isinstance(storage_obj, DataFlowStorage), f"Storage must be a DataFlowStorage object, but got {type(storage_obj)} in {op_runtime}'s `run` function with key `storage`."
            
            # create an operator node
            op_node = OperatorNode(
                op_obj=op_runtime.op,
                op_name=op_runtime.op_name,
                storage=storage_obj,
                llm_serving=llm_serving_obj,
                **op_runtime.kwargs
            )
            
            # append to lists, if None, just keep it
            self.op_nodes_list.append(op_node)
            self.llm_serving_list.append(llm_serving_obj)
            if llm_serving_obj is not None:
                self.llm_serving_counter[llm_serving_obj] += 1
        self.logger.debug(f"Built operator nodes graph with {self.op_nodes_list} nodes, \nand {self.llm_serving_list} LLM Serving objects.")
        
        # get keys in the first storage:
        first_op = self.op_nodes_list[0] if self.op_nodes_list else None
        if first_op and first_op.storage:
            iter_storage_keys = first_op.storage.get_keys_from_dataframe()
        else:
            iter_storage_keys = []

        # print("start keys", iter_storage_keys)

        # all keys in the first storage will be the initial keys for validation
        self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))

        # build graph of all operators and keys from all states
        for op_node in self.op_nodes_list:
            # check if accumulated_keys have the input keys of this operator
            # print(op_node, op_node.input_keys, op_node.output_keys)
            for input_key in op_node.input_keys:
                if input_key not in self.accumulated_keys[-1]:
                    error_msg = (
                        f"Input key '{input_key}' in `{op_node.op_name}` does not match any output keys from previous operators "
                        f"or any dataset keys. Please check parameter "
                        f"'{op_node.input_key_nodes[input_key].key_para_name}' in the `run()` of the operator "
                        f"'{op_node.op_name}' (class '{op_node.op_obj.__class__.__name__}')."
                    )
                    self.logger.warning(error_msg)
                    raise KeyError(error_msg)
            # add output keys to accumulated keys
            for output_key in op_node.output_keys:
                if output_key not in iter_storage_keys:
                    iter_storage_keys.append(output_key)
            self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))
        self.final_keys = copy.deepcopy(iter_storage_keys)
        
        for i, keys in enumerate(self.accumulated_keys):
            # print(i, keys)
            pass
        self.logger.debug(f"Accumulated keys after building graph: {self.accumulated_keys}")
    
        self.dataset_node = OperatorNode(
            None,
            "THEDATASET",
            None,
            None,

        )
        self.dataset_node.init_dataset_node(self.accumulated_keys[0])
        self.op_nodes_list.insert(0, self.dataset_node)

        # set a default dict for all keys
        self.last_modified_index_of_keys: dict[list] = {}
        for key in self.final_keys:
            self.last_modified_index_of_keys[key] = []
        # print(self.last_modified_index_of_keys)

        # now the first op node is THEDATASET op
        for idx, i_op in enumerate(self.op_nodes_list):
            # check for input keys
            for input_key in i_op.input_keys:
                current_keynode:KeyNode = i_op.input_key_nodes[input_key]
                current_keynode.set_index(idx)

                if len(self.last_modified_index_of_keys[input_key]) > 0:
                    last_modified_idx = self.last_modified_index_of_keys[input_key][-1]
                    last_modified_keynode:KeyNode = self.op_nodes_list[last_modified_idx].output_keys_nodes[input_key]
                    # double side ptr for each nodes
                    last_modified_keynode.ptr.append(current_keynode)
                    current_keynode.ptr.append(last_modified_keynode)
            # check for output keys
            for output_key in i_op.output_keys:
                current_keynode:KeyNode = i_op.output_keys_nodes[output_key]
                current_keynode.set_index(idx)
                self.last_modified_index_of_keys[output_key].append(idx)

        for key, value in self.last_modified_index_of_keys.items():
            # print(key, value)
            pass
                    
        for op in self.op_nodes_list:
            # print(op)
            pass
            
    # def _build_serving_resources_map(self):
    #     for op_runtime in self.op_runtimes:
    #         for _, v in vars(op_runtime.op).items():
    #             if isinstance(v, LLMServingABC):
    #                 self.serving_resources[op_runtime.op]["LLMServingABC"] = v
    #                 self.serving_reference_count[v] += 1
                    
    def _compiled_forward(self):
        # for loop for each op and its `storage` status
        for op_node in self.op_nodes_list:
            self.logger.debug(f"Ready to run {op_node}, with serving={op_node.llm_serving}, active_llm_serving={self.active_llm_serving}")
            if op_node.llm_serving != None:
                if self.active_llm_serving and self.active_llm_serving is not op_node.llm_serving:
                    self.logger.debug(f"Detected active LLM Serving {self.active_llm_serving}, new serving {op_node.llm_serving}, cleaning up...")
                    self.active_llm_serving.cleanup()
                self.active_llm_serving = op_node.llm_serving

                op_node.op_obj.run(
                    storage=op_node.storage,
                    **op_node.kwargs
                )
            if op_node.llm_serving != None:
                self.llm_serving_counter[self.active_llm_serving] -= 1
                if self.llm_serving_counter[self.active_llm_serving] == 0:
                    self.logger.debug(f"Detected LLM Serving {self.active_llm_serving} ref reduced to 0, cleaning up...")
                    self.active_llm_serving.cleanup()
                    self.active_llm_serving = None
