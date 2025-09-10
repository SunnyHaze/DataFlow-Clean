import os
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import re
from dataflow.prompts.text2sql import SelectVecSQLGeneratorPrompt
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class VecSQLGenerator(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC, 
                 database_manager: DatabaseManager,
                 generate_num: int = 10,
                 prompt_template = None
        ):
        self.llm_serving = llm_serving
        self.logger = get_logger()
        self.database_manager = database_manager
        self.generate_num = generate_num
        if prompt_template is None:
            self.prompt_template = SelectVecSQLGeneratorPrompt()
        else:
            self.prompt_template = prompt_template
        random.seed(42)

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "基于数据库信息，合成VecSQL，覆盖不同的难度、数据库Schema、函数和风格。\n\n"
                "输出参数：\n"
                "- output_sql_key: 输出SQL列名\n"
                "- output_db_id_key: 数据库ID列名\n\n"
            )
        elif lang == "en":
            return (
                "This operator synthesizes VecSQL based on database information, covering different complexities, schemas, functions, and styles.\n\n"
                "Output parameters:\n"
                "- output_sql_key: The name of the output SQL column\n"
                "- output_db_id_key: The name of the database ID column\n\n"
            )
        else:
            return "VecSQL generator for Text2VecSQL tasks."

    def get_create_statements_and_insert_statements(self, db_id: str) -> str:
        return self.database_manager.get_create_statements_and_insert_statements(db_id)

    def parse_response(self, response):
        if not response:
            return ""  
        pattern = r"```sql\s*(.*?)\s*```"
        sql_blocks = re.findall(pattern, response, re.DOTALL)
            
        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            self.logger.warning("No SQL code block found in the response")
            return ""

    def run(self, storage: DataFlowStorage,
            output_sql_key: str = "sql",
            output_db_id_key: str = "db_id"
        ):
        self.output_sql_key = output_sql_key
        self.output_db_id_key = output_db_id_key
        raw_dataframe = storage.read("dataframe")
        
        db_names = self.database_manager.list_databases()
        prompts = []
        self.logger.info(f"Generating {self.generate_num} VecSQLs for each database")

        for db_name in tqdm(db_names, desc="Processing Databases"):
            create_statements, insert_statements = self.get_create_statements_and_insert_statements(db_name)

            for _ in range(self.generate_num):
                # Unpack the tuple here, taking only the first element (the prompt string)
                prompt, _ = self.prompt_template.build_prompt(
                    insert_statements=insert_statements,
                    create_statements=create_statements,
                    db_engine=self.database_manager.db_type
                )
                prompts.append({"prompt": prompt, "db_id": db_name})
            
        if not prompts:
            self.logger.warning("No prompts generated, please check the database path and file")
            return [self.output_sql_key, self.output_db_id_key]
            
        db_ids = [data["db_id"] for data in prompts]
        prompt_list = [data["prompt"] for data in prompts]
        
        try:
            responses = self.llm_serving.generate_from_input(prompt_list, "")
        except Exception as e:
            self.logger.error(f"Failed to generate SQLs: {e}")
            responses = [""] * len(prompt_list)
            
        results = [
            {
                self.output_db_id_key: db_id,
                self.output_sql_key: self.parse_response(response)
            }
            for db_id, response in zip(db_ids, responses)
        ]
        
        output_file = storage.write(pd.DataFrame(results))
        return [self.output_sql_key, self.output_db_id_key]
