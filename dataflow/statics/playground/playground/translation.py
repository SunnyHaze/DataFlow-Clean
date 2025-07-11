from dataflow.operators.generate.GeneralText import PromptGenerator
from dataflow.serving import LocalModelLLMServing, APILLMServing_request
from dataflow.utils.storage import FileStorage

class GPT_generator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/translation.jsonl",
            cache_path="./cache",
            file_name_prefix="translation",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=2
        )
        self.prompt_generator = PromptGenerator(llm_serving = self.llm_serving)        

    def forward(self):
        # Initial filters
        self.prompt_generator.run(
            storage = self.storage.step(),
            system_prompt = "Please translate to Chinese.",
            input_key = "raw_content",
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = GPT_generator()
    model.forward()
