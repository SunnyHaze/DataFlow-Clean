import json
import os
import random
import sqlite3
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

from dataflow.prompts.text2sql import QuestionGenerationPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class QuestionGeneration(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, database_manager: DatabaseManager, question_candidates_num: int = 5):
        self.llm_serving = llm_serving
        self.database_manager = database_manager
        self.prompt = QuestionGenerationPrompt()
        self.logger = get_logger()
        self.question_candidates_num = question_candidates_num
        random.seed(42)

    def extract_column_descriptions(self, create_statements):
        column_name2column_desc = dict()
        pattern = r'"(\w+)"\s+\w+\s*/\*\s*(.*?)\s*\*/'

        for create_statement in create_statements:
            matches = re.findall(pattern, create_statement)

            for column_name, description in matches:
                column_name = column_name.lower()
                if column_name not in column_name2column_desc:
                    column_name2column_desc[column_name] = description

        return column_name2column_desc

    def parse_llm_response(self, response, style):
        explanation_pattern = re.compile(r'\[EXPLANATION-START\](.*?)\[EXPLANATION-END\]', re.DOTALL)
        question_pattern = re.compile(r'\[QUESTION-START\](.*?)\[QUESTION-END\]', re.DOTALL)
        external_knowledge_pattern = re.compile(r'\[EXTERNAL-KNOWLEDGE-START\](.*?)\[EXTERNAL-KNOWLEDGE-END\]', re.DOTALL)

        explanation_match = explanation_pattern.search(response)
        question_match = question_pattern.search(response)
        external_knowledge_match = external_knowledge_pattern.search(response)

        explanation_content = explanation_match.group(1).strip() if explanation_match else ""
        question_content = question_match.group(1).strip() if question_match else ""
        external_knowledge_content = external_knowledge_match.group(1).strip() if external_knowledge_match else ""

        if explanation_content == "" or question_content == "":
            return None
        else:
            return {
                "question": question_content.strip(),
                "external_knowledge": external_knowledge_content.strip()
            }

    def select_best_question(self, question_candidates, embedding_model):
        if len(question_candidates) == 0:
            return None
        elif len(question_candidates) == 1:
            return question_candidates[0]
        elif len(question_candidates) == 2:
            return random.sample(question_candidates, 1)[0]
        else:
            texts = [question_info["external_knowledge"] + " " + question_info["question"] for question_info in question_candidates]
            texts = [text.strip() for text in texts]
            embeddings = embedding_model.encode(texts)
            distance_matrix = cdist(embeddings, embeddings, metric='cosine')
            distance_sums = distance_matrix.sum(axis=1)
            min_index = np.argmin(distance_sums)
            return question_candidates[min_index]

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "sql",
            input_db_id_key: str = "db_id",
            output_question_key: str = "question"
        ):
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.output_question_key = output_question_key
        
        raw_dataframe = storage.read("dataframe")
        raw_data = [row for _, row in raw_dataframe.iterrows()]
        
        styles = ["Formal", "Colloquial", "Imperative", "Interrogative", "Descriptive", "Concise", "Vague", "Metaphorical"]
        db_ids = list(set([data[self.input_db_id_key] for data in raw_data]))
        db_id2column_info = dict()
        style2desc = self.prompt.get_style2desc()
        
        for db_id in tqdm(db_ids, desc="Extracting database schema"):
            _, create_statements = self.database_manager.get_table_names_and_create_statements(db_id)
            db_id2column_info[db_id] = self.extract_column_descriptions(create_statements)
        
        self.logger.info("Generating question candidates...")
        prompts = []
        prompt_data_mapping = []
        
        for data in tqdm(raw_data, desc="Preparing prompts"):
            style_name = random.sample(styles, 1)[0]
            column_name2column_desc = db_id2column_info[data[self.input_db_id_key]]
            used_column_name2column_desc = dict()
            
            for column_name, column_desc in column_name2column_desc.items():
                if column_name.lower() in data[self.input_sql_key].lower():
                    used_column_name2column_desc[column_name] = column_desc

            if style_name in ["Vague", "Metaphorical"]:
                steps = self.prompt.get_steps_w_ek()
                guidelines = self.prompt.get_guidelines_w_ek()
                instruction = self.prompt.get_instruction_w_ek()
                output_format = self.prompt.get_output_format_w_ek()
            else:
                steps = self.prompt.get_steps_wo_ek()
                guidelines = self.prompt.get_guidelines_wo_ek()
                instruction = self.prompt.get_instruction_wo_ek()
                output_format = self.prompt.get_output_format_wo_ek()

            prompt = self.prompt.question_synthesis_prompt(
                style_desc=style2desc[style_name].strip(),
                engine=self.database_manager.db_type,
                column_info=json.dumps(used_column_name2column_desc, indent=2, ensure_ascii=False).strip(),
                sql=data[self.input_sql_key].strip(),
                steps=steps.strip(),
                guidelines=guidelines.strip(),
                output_format=output_format.strip(),
                instruction=instruction.strip()
            )
            
            for _ in range(self.question_candidates_num):
                prompts.append(prompt)
                prompt_data_mapping.append({**data, "style": style_name})

        responses = self.llm_serving.generate_from_input(prompts, system_prompt="You are a helpful assistant.")
        
        self.logger.info("Parsing responses and organizing candidates...")
        grouped_responses = [responses[i:i+self.question_candidates_num] for i in range(0, len(responses), self.question_candidates_num)]
        
        embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/all-mpnet-base-v2", device="cuda:0")
        
        results = []
        for data, response_group in zip(raw_data, grouped_responses):
            question_candidates = []
            for response in response_group:
                parsed_response = self.parse_llm_response(response, data.get("style", "Formal"))
                if parsed_response:
                    question_candidates.append(parsed_response)
            
            best_question = self.select_best_question(question_candidates, embedding_model)
            
            if best_question:
                result = {
                    **data,
                    self.output_question_key: best_question["question"]
                }
                results.append(result)
            else:
                self.logger.warning(f"No valid question generated for data: {data[self.input_db_id_key]}")

        output_file = storage.write(pd.DataFrame(results))
        self.logger.info(f"Question generation results saved to {output_file}")
        
        return [self.output_question_key]