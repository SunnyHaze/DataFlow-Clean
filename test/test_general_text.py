from dataflow.operators.conversations import ConsistentChatGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request 


class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o"
        )
        self.model_cache_dir = './dataflow_cache'
        self.processor = ConsistentChatGenerator(llm_serving=serving, num_dialogs_per_intent=1)

    def forward(self):
        self.processor.run(
            storage=self.storage.step()
        )

model = TextPipeline()
model.forward()