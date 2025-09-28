import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
from typing import List

# 假设以下模块已在您的项目中定义好
from dataflow.core import LLMServingABC
from ..logger import get_logger


class LocalEmbeddingServing(LLMServingABC):
    """
    使用本地 sentence-transformer 模型生成嵌入。
    接口风格与 APILLMServing_request 对齐，并通过 max_workers 支持并行计算。
    """
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 max_workers: int = 2,
                 max_retries: int = 3
                 ):
        """
        初始化本地嵌入模型。

        Args:
            model_name (str): 要加载的 sentence-transformer 模型名称。
            device (str): 优先使用的设备类型 ('cpu' 或 'cuda')。若为 None，将自动检测。
            max_workers (int): 并行工作单元的数量。
                               - 1 (默认): 不使用并行，在单个设备上运行。
                               - > 1: 启用并行模式。
                                 - 在CPU上: 将启用多进程模式（使用所有可用核心）。
                                 - 在多GPU上: 将使用 min(max_workers, 本机GPU数) 个GPU进行数据并行。
            max_retries (int): 在发生错误时，最大重试次数。
        """
        self.logger = get_logger()
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

        self._execution_strategy = "single_device"
        self._target_devices = None

        # 1. 确定设备类型和并行策略
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.max_workers > 1:
            if 'cpu' in device:
                cpu_count = os.cpu_count() or 1
                if cpu_count > 1:
                    self._execution_strategy = "cpu_parallel"
                    self.logger.info(f"CPU并行模式已启用 (max_workers > 1)。")
                    self.logger.warning(f"注意: sentence-transformers将使用所有 {cpu_count} 个可用CPU核心，"
                                        "max_workers参数仅用于启用此模式，不限制核心数。")
                else:
                    self.logger.warning("max_workers > 1 但仅检测到1个CPU核心，回退到单设备模式。")

            elif 'cuda' in device:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    # 【采纳您的建议】取 max_workers 和实际GPU数量的最小值
                    num_gpus_to_use = min(self.max_workers, gpu_count)
                    
                    if num_gpus_to_use > 1:
                        self._execution_strategy = "gpu_parallel"
                        self._target_devices = [f'cuda:{i}' for i in range(num_gpus_to_use)]
                        self.logger.info(f"多GPU并行模式已启用。将使用 {num_gpus_to_use} 张GPU (min({self.max_workers}, {gpu_count}))。")
                        self.logger.info(f"目标设备: {self._target_devices}")
                    else:
                        self.logger.warning("max_workers > 1 但最终计算使用的GPU数量为1，回退到单设备模式。")
                else:
                    self.logger.warning("max_workers > 1 但仅检测到1张GPU，回退到单设备模式。")

        # 2. 加载模型到主设备
        if self._execution_strategy == "gpu_parallel":
            self.primary_device = self._target_devices[0]
        else:
            self.primary_device = 'cuda:0' if 'cuda' in device and torch.cuda.is_available() else 'cpu'

        self.logger.info(f"正在加载模型 '{self.model_name}' 到主设备 '{self.primary_device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.primary_device)
        self.logger.info("模型加载成功。")

    def start_serving(self) -> None:
        self.logger.info("LocalEmbeddingServing: 无需启动独立服务，模型已在内存中。")

    def generate_embedding_from_input(self,
                                      texts: List[str],
                                      batch_size: int = 32
                                      ) -> List[List[float]]:
        """
        为文本列表生成嵌入，包含重试逻辑和并行执行逻辑。
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"开始为 {len(texts)} 条文本生成嵌入 (尝试 {attempt + 1}/{self.max_retries})...")

                if self._execution_strategy == "gpu_parallel":
                    pool = self.model.start_multi_process_pool(target_devices=self._target_devices)
                    embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=batch_size)
                    self.model.stop_multi_process_pool(pool)

                elif self._execution_strategy == "cpu_parallel":
                    pool = self.model.start_multi_process_pool()
                    embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=batch_size)
                    self.model.stop_multi_process_pool(pool)

                else: # 单设备模式
                    embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

                self.logger.info("嵌入生成成功。")
                return embeddings.tolist()

            except Exception as e:
                last_exception = e
                self.logger.error(f"生成嵌入时发生错误 (尝试 {attempt + 1}): {e}")
                if attempt + 1 < self.max_retries:
                    wait_time = 2 ** attempt
                    self.logger.info(f"将在 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("所有重试均失败。")
                    raise last_exception

        return []

    def cleanup(self):
        self.logger.info(f"清理模型 '{self.model_name}' 的资源...")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("清理完成。")

    # --- 其他接口的适配 ---
    def generate_from_input(self, user_inputs: List[str], system_prompt: str = "") -> List[str]:
        self.logger.warning("generate_from_input 不适用于 LocalEmbeddingServing。")
        return [None] * len(user_inputs)

    def generate_from_conversations(self, conversations: List[List[dict]]) -> List[str]:
        self.logger.warning("generate_from_conversations 不适用于 LocalEmbeddingServing。")
        return [None] * len(conversations)
