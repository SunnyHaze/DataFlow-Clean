from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
import sqlite3
import pymysql
import threading
import os
import glob
from dataclasses import dataclass
from dataflow import get_logger
from tqdm import tqdm
from queue import Queue, Empty
from contextlib import contextmanager
import time
import asyncio
import concurrent.futures
import hashlib
from enum import Enum
import sys

class OperationType(Enum):
    EXECUTE_QUERY = "execute_query"
    ANALYZE_PLAN = "analyze_plan"
    COMPARE_SQL = "compare_sql"
    VALIDATE_SQL = "validate_sql"

@dataclass
class DatabaseInfo:
    db_id: str
    db_type: str
    connection_info: Dict
    metadata: Optional[Dict] = None

@dataclass
class BatchOperation:
    operation_type: OperationType
    db_id: str
    sql: str
    timeout: float = 2.0
    additional_params: Optional[Dict] = None
    operation_id: Optional[str] = None

@dataclass
class BatchResult:
    operation_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

class DatabaseConnector(ABC):
    
    @abstractmethod
    def connect(self, connection_info: Dict) -> Any:
        pass

    @abstractmethod
    def get_execution_plan(self, connection, sql: str) -> List[Any]:
        pass
    
    @abstractmethod
    def execute_query(self, connection, sql: str) -> List:
        pass
    
    @abstractmethod
    def get_tables(self, connection) -> List[str]:
        pass
    
    @abstractmethod
    def get_table_schema(self, connection, table_name: str) -> Dict:
        pass
    
    @abstractmethod
    def get_sample_data(self, connection, table_name: str, limit: int) -> Dict[str, Any]:
        pass

class SQLiteConnector(DatabaseConnector):
    
    def connect(self, connection_info: Dict) -> sqlite3.Connection:
        db_path = connection_info['path']
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
    
    def get_execution_plan(self, connection: sqlite3.Connection, sql: str) -> List[Any]:
        cursor = connection.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        return cursor.fetchall()

    def execute_query(self, connection: sqlite3.Connection, sql: str) -> List:
        cursor = connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def get_tables(self, connection: sqlite3.Connection) -> List[str]:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_schema(self, connection: sqlite3.Connection, table_name: str) -> Dict:
        cursor = connection.cursor()
        
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = cursor.fetchall()
        
        schema = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': []
        }
        
        for col_info in columns:
            col_name = col_info[1]
            col_type = col_info[2]
            is_pk = col_info[5]
            
            schema['columns'][col_name] = {
                'type': self._normalize_type(col_type),
                'raw_type': col_type
            }
            
            if is_pk:
                schema['primary_keys'].append(col_name)
        
        cursor.execute(f"PRAGMA foreign_key_list([{table_name}])")
        foreign_keys = cursor.fetchall()
        
        for fk in foreign_keys:
            schema['foreign_keys'].append({
                'column': fk[3],
                'referenced_table': fk[2],
                'referenced_column': fk[4]
            })
        
        return schema
    
    def get_sample_data(self, connection: sqlite3.Connection, table_name: str, limit: int) -> Dict[str, Any]:
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
        rows = cursor.fetchall()
        
        column_names = [description[0] for description in cursor.description]
        
        return {
            'columns': column_names,
            'rows': rows
        }
    
    def _normalize_type(self, db_type: str) -> str:
        if not db_type:
            return 'text'
        
        db_type = db_type.lower()
        
        if any(t in db_type for t in ['int', 'integer', 'number']):
            return 'integer'
        elif any(t in db_type for t in ['real', 'float', 'double']):
            return 'real'
        elif any(t in db_type for t in ['text', 'varchar', 'char']):
            return 'text'
        elif any(t in db_type for t in ['blob']):
            return 'blob'
        else:
            return 'text'

class MySQLConnector(DatabaseConnector):
    
    def connect(self, connection_info: Dict) -> pymysql.Connection:
        return pymysql.connect(**connection_info, connect_timeout=2)

    def get_execution_plan(self, connection: pymysql.Connection, sql: str) -> List[Any]:
        cursor = connection.cursor()
        cursor.execute(f"EXPLAIN {sql}")
        return cursor.fetchall()
    
    def execute_query(self, connection: pymysql.Connection, sql: str) -> List:
        cursor = connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def get_tables(self, connection: pymysql.Connection) -> List[str]:
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_schema(self, connection: pymysql.Connection, table_name: str) -> Dict:
        cursor = connection.cursor()
        database = connection.db.decode('utf-8')
        
        cursor.execute("""
            SELECT column_name, data_type, column_key, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (database, table_name))
        columns = cursor.fetchall()
        
        schema = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': []
        }
        
        for col_name, col_type, col_key, is_nullable, col_default in columns:
            schema['columns'][col_name] = {
                'type': self._normalize_type(col_type),
                'raw_type': col_type,
                'nullable': is_nullable == 'YES',
                'default': col_default
            }
            
            if col_key == 'PRI':
                schema['primary_keys'].append(col_name)
        
        cursor.execute("""
            SELECT column_name, referenced_table_name, referenced_column_name
            FROM information_schema.key_column_usage 
            WHERE table_schema = %s AND table_name = %s
            AND referenced_table_name IS NOT NULL
        """, (database, table_name))
        foreign_keys = cursor.fetchall()
        
        for col_name, ref_table, ref_column in foreign_keys:
            schema['foreign_keys'].append({
                'column': col_name,
                'referenced_table': ref_table,
                'referenced_column': ref_column
            })
        
        return schema
    
    def get_sample_data(self, connection: pymysql.Connection, table_name: str, limit: int) -> Dict[str, Any]:
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM `{table_name}` LIMIT {limit}')
        rows = cursor.fetchall()
        
        column_names = [description[0] for description in cursor.description]
        
        return {
            'columns': column_names,
            'rows': rows
        }
    
    def _normalize_type(self, db_type: str) -> str:
        if not db_type:
            return 'text'
        
        db_type = db_type.lower()
        
        type_mapping = {
            'int': 'integer', 'integer': 'integer', 'bigint': 'integer', 'smallint': 'integer',
            'tinyint': 'integer', 'mediumint': 'integer',
            'float': 'real', 'double': 'real', 'decimal': 'real', 'numeric': 'real',
            'varchar': 'text', 'char': 'text', 'text': 'text', 'longtext': 'text',
            'mediumtext': 'text', 'tinytext': 'text',
            'date': 'date', 'datetime': 'datetime', 'timestamp': 'datetime', 'time': 'time',
            'blob': 'blob', 'longblob': 'blob', 'mediumblob': 'blob', 'tinyblob': 'blob'
        }
        
        return type_mapping.get(db_type, 'text')

class DatabaseRegistry:
    
    def __init__(self, logger=None):
        self.logger = logger
        self._registry: Dict[str, DatabaseInfo] = {}
    
    def register_database(self, db_info: DatabaseInfo):
        self._registry[db_info.db_id] = db_info
    
    def get_database(self, db_id: str) -> Optional[DatabaseInfo]:
        return self._registry.get(db_id)
    
    def list_databases(self) -> List[str]:
        return list(self._registry.keys())
    
    def database_exists(self, db_id: str) -> bool:
        return db_id in self._registry
    
    def clear(self):
        self._registry.clear()

class CacheManager:
    
    def __init__(self, max_size: int = 2000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._max_similar_queries = 10 
        
        self._schema_cache: Dict[str, Dict] = {}
        self._query_result_cache: Dict[str, Dict] = {}
        self._execution_plan_cache: Dict[str, List] = {}
        self._table_info_cache: Dict[str, Dict] = {}
        
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()

        self._query_pattern_cache: Dict[str, Dict] = {}
        self._similar_query_cache: Dict[str, List[Dict]] = {}
        
        self._hit_counts = {
            'schema': 0, 'query': 0, 'plan': 0, 'table': 0
        }
        self._total_requests = {
            'schema': 0, 'query': 0, 'plan': 0, 'table': 0
        }

    def _normalize_sql(self, sql: str) -> str:
        """规范化SQL以识别相似查询"""
        import re
        sql = re.sub(r'\s+', ' ', sql.strip().lower())
        sql = re.sub(r'--.*?\n', '', sql)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql

    def get_query_result_smart(self, db_id: str, sql: str) -> Optional[Dict]:
        # 先尝试精确匹配
        exact_result = self.get_query_result(db_id, sql)
        if exact_result:
            return exact_result
        
        # 尝试相似查询匹配
        normalized_sql = self._normalize_sql(sql)
        cache_key = self._generate_cache_key("similar", db_id, normalized_sql)
        
        with self._lock:
            if cache_key in self._similar_query_cache:
                for item in self._similar_query_cache[cache_key]:
                    if item['normalized'] == normalized_sql:
                        similar_result = self.get_query_result(db_id, item['original'])
                        if similar_result:
                            return similar_result
        
        return None
    
    def set_query_result_smart(self, db_id: str, sql: str, result: Dict):
        """智能设置查询结果，限制相似查询数量"""
        self.set_query_result(db_id, sql, result)
        
        normalized_sql = self._normalize_sql(sql)
        cache_key = self._generate_cache_key("similar", db_id, normalized_sql)
        
        with self._lock:
            if cache_key not in self._similar_query_cache:
                self._similar_query_cache[cache_key] = []
            
            # 检查是否已存在相同的规范化SQL
            existing = any(item['normalized'] == normalized_sql 
                          for item in self._similar_query_cache[cache_key])
            
            if not existing:
                self._similar_query_cache[cache_key].append({
                    'normalized': normalized_sql,
                    'original': sql
                })
                
                # 限制数量
                if len(self._similar_query_cache[cache_key]) > self._max_similar_queries:
                    self._similar_query_cache[cache_key].pop(0)
        
    def _generate_cache_key(self, prefix: str, *args) -> str:
        key_parts = [prefix] + [str(arg).replace('|', '\\|') for arg in args]
        key_string = '|'.join(key_parts)
        return f"{prefix}:{hashlib.md5(key_string.encode()).hexdigest()[:16]}"
    
    def _safe_deep_copy(self, data: Any) -> Any:
        """安全的深拷贝，处理各种数据类型"""
        try:
            if isinstance(data, dict):
                return {k: self._safe_deep_copy(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._safe_deep_copy(item) for item in data]
            elif isinstance(data, tuple):
                return tuple(self._safe_deep_copy(item) for item in data)
            elif isinstance(data, (str, int, float, bool, type(None))):
                return data
            else:
                # 对于其他类型，尝试使用copy.deepcopy
                import copy
                return copy.deepcopy(data)
        except Exception:
            # 如果深拷贝失败，返回原对象
            return data
    
    def _get_from_cache(self, cache_dict: Dict, cache_key: str, cache_type: str) -> Optional[Any]:
        """通用缓存获取方法"""
        with self._lock:
            self._total_requests[cache_type] += 1
            
            if cache_key in cache_dict:
                if time.time() - self._access_times.get(cache_key, 0) > self.ttl:
                    self._remove_from_cache(cache_dict, cache_key)
                    return None
                
                self._access_times[cache_key] = time.time()
                self._hit_counts[cache_type] += 1
                
                # 使用安全的深拷贝
                return self._safe_deep_copy(cache_dict[cache_key])
            return None
    
    def _set_to_cache(self, cache_dict: Dict, cache_key: str, data: Any):
        """通用缓存设置方法"""
        with self._lock:
            if len(cache_dict) >= self.max_size // 4:
                oldest_keys = sorted(
                    [k for k in cache_dict.keys() if k in self._access_times],
                    key=lambda k: self._access_times[k]
                )[:len(cache_dict) // 4]
                
                for old_key in oldest_keys:
                    self._remove_from_cache(cache_dict, old_key)
            
            # 使用安全的深拷贝存储数据
            cache_dict[cache_key] = self._safe_deep_copy(data)
            self._access_times[cache_key] = time.time()
    
    def _remove_from_cache(self, cache_dict: Dict, cache_key: str):
        """从缓存中移除"""
        cache_dict.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
    
    # Schema 缓存
    def get_schema(self, db_id: str, db_type: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key("schema", db_id, db_type)
        return self._get_from_cache(self._schema_cache, cache_key, 'schema')
    
    def set_schema(self, db_id: str, db_type: str, schema: Dict):
        cache_key = self._generate_cache_key("schema", db_id, db_type)
        self._set_to_cache(self._schema_cache, cache_key, schema)
    
    # 查询结果缓存
    def get_query_result(self, db_id: str, sql: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key("query", db_id, sql)
        return self._get_from_cache(self._query_result_cache, cache_key, 'query')
    
    def set_query_result(self, db_id: str, sql: str, result: Dict):
        cache_key = self._generate_cache_key("query", db_id, sql)
        self._set_to_cache(self._query_result_cache, cache_key, result)
    
    # 执行计划缓存
    def get_execution_plan(self, db_id: str, sql: str) -> Optional[List]:
        cache_key = self._generate_cache_key("plan", db_id, sql)
        return self._get_from_cache(self._execution_plan_cache, cache_key, 'plan')
    
    def set_execution_plan(self, db_id: str, sql: str, plan: List):
        cache_key = self._generate_cache_key("plan", db_id, sql)
        self._set_to_cache(self._execution_plan_cache, cache_key, plan)
    
    # 表信息缓存
    def get_table_info(self, db_id: str, table_name: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key("table", db_id, table_name)
        return self._get_from_cache(self._table_info_cache, cache_key, 'table')
    
    def set_table_info(self, db_id: str, table_name: str, info: Dict):
        cache_key = self._generate_cache_key("table", db_id, table_name)
        self._set_to_cache(self._table_info_cache, cache_key, info)
    
    def clear_all(self):
        """清空所有缓存"""
        with self._lock:
            self._schema_cache.clear()
            self._query_result_cache.clear()
            self._execution_plan_cache.clear()
            self._table_info_cache.clear()
            self._query_pattern_cache.clear()
            self._similar_query_cache.clear()
            self._access_times.clear()
            
            for key in self._hit_counts:
                self._hit_counts[key] = 0
                self._total_requests[key] = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_size = (len(self._schema_cache) + len(self._query_result_cache) + 
                        len(self._execution_plan_cache) + len(self._table_info_cache) +
                        len(self._query_pattern_cache) + len(self._similar_query_cache))
            
            hit_rates = {}
            for cache_type in self._hit_counts:
                total_req = self._total_requests[cache_type]
                hit_rates[cache_type] = self._hit_counts[cache_type] / max(total_req, 1)
            
            return {
                'total_size': total_size,
                'max_size': self.max_size,
                'cache_sizes': {
                    'schema': len(self._schema_cache),
                    'query': len(self._query_result_cache),
                    'plan': len(self._execution_plan_cache),
                    'table': len(self._table_info_cache),
                    'pattern': len(self._query_pattern_cache),
                    'similar': len(self._similar_query_cache)
                },
                'hit_rates': hit_rates,
                'memory_usage_estimate': self._estimate_memory_usage()
            }

    def _estimate_memory_usage(self) -> Dict[str, int]:
        """估算内存使用量"""
        def get_size(obj):
            try:
                if isinstance(obj, dict):
                    return sum(sys.getsizeof(k) + get_size(v) for k, v in obj.items()) + sys.getsizeof(obj)
                elif isinstance(obj, (list, tuple)):
                    return sum(get_size(item) for item in obj) + sys.getsizeof(obj)
                else:
                    return sys.getsizeof(obj)
            except:
                return sys.getsizeof(obj)
            
        with self._lock:
            return {
                'schema_cache': sum(get_size(v) for v in self._schema_cache.values()),
                'query_cache': sum(get_size(v) for v in self._query_result_cache.values()),
                'plan_cache': sum(get_size(v) for v in self._execution_plan_cache.values()),
                'table_cache': sum(get_size(v) for v in self._table_info_cache.values()),
                'pattern_cache': sum(get_size(v) for v in self._query_pattern_cache.values()),
                'similar_cache': sum(get_size(v) for v in self._similar_query_cache.values()),
                'access_times': get_size(self._access_times)
            }

class ConnectionPool:
    def __init__(self, connector, connection_info, max_connections=10, max_idle_time=300):
        self.connector = connector
        self.connection_info = connection_info
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        self._pool = Queue(maxsize=max_connections)
        self._active_connections = 0
        self._lock = threading.Lock()
        self._connection_times = {}

    @contextmanager
    def get_connection(self):
        conn = None
        conn_created = False
        try:
            # 尝试从池中获取连接
            try:
                conn_info = self._pool.get_nowait()
                conn = conn_info['connection']
                if not self._is_connection_valid(conn):
                    self._close_connection_safely(conn)
                    conn = None
            except Empty:
                pass
            
            # 如果没有可用连接，创建新连接
            if conn is None:
                with self._lock:
                    if self._active_connections < self.max_connections:
                        try:
                            conn = self.connector.connect(self.connection_info)
                            self._active_connections += 1
                            conn_created = True
                        except Exception as e:
                            raise Exception(f"Failed to create connection: {str(e)}")
                    else:
                        # 等待可用连接
                        try:
                            conn_info = self._pool.get(timeout=5)
                            conn = conn_info['connection']
                            if not self._is_connection_valid(conn):
                                self._close_connection_safely(conn)
                                raise Exception("No valid connections available")
                        except Empty:
                            raise Exception("Connection pool exhausted, timeout waiting for connection")
            
            yield conn
            
        except Exception as e:
            # 如果连接有问题，不要放回池中
            if conn and not self._is_connection_valid(conn):
                self._close_connection_safely(conn)
                if conn_created:
                    with self._lock:
                        self._active_connections -= 1
                conn = None
            raise e
        finally:
            if conn:
                try:
                    # 连接正常，放回池中
                    self._pool.put_nowait({
                        'connection': conn,
                        'last_used': time.time()
                    })
                except:
                    # 池已满，关闭连接
                    self._close_connection_safely(conn)
                    if conn_created:
                        with self._lock:
                            self._active_connections -= 1
    
    def _is_connection_valid(self, conn, timeout=2.0) -> bool:
        """检查连接是否有效，添加超时机制"""
        if conn is None:
            return False
        
        def check_conn():
            try:
                # 不同数据库的连接检查方法
                if hasattr(conn, 'ping'):
                    # MySQL连接
                    conn.ping(reconnect=False)
                    return True
                elif hasattr(conn, 'execute'):
                    # SQLite连接
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return True
                else:
                    # 通用检查
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return True
            except Exception:
                return False
        
        # 使用线程和超时来检查连接
        result = [False]
        
        def check_thread():
            result[0] = check_conn()
        
        thread = threading.Thread(target=check_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # 超时，认为连接无效
            return False
        
        return result[0]
    
    def _close_connection_safely(self, conn):
        """安全关闭连接"""
        try:
            if conn:
                conn.close()
        except Exception:
            pass  # 忽略关闭时的异常
    
    def cleanup_idle_connections(self):
        """清理空闲连接"""
        current_time = time.time()
        active_connections = []
        
        while True:
            try:
                conn_info = self._pool.get_nowait()
                if current_time - conn_info['last_used'] > self.max_idle_time:
                    self._close_connection_safely(conn_info['connection'])
                    with self._lock:
                        self._active_connections -= 1
                else:
                    active_connections.append(conn_info)
            except Empty:
                break
        
        # 将仍然活跃的连接放回池中
        for conn_info in active_connections:
            try:
                self._pool.put_nowait(conn_info)
            except:
                # 如果放不回去，关闭连接
                self._close_connection_safely(conn_info['connection'])
                with self._lock:
                    self._active_connections -= 1
    
    def close_all(self):
        """关闭所有连接"""
        while True:
            try:
                conn_info = self._pool.get_nowait()
                self._close_connection_safely(conn_info['connection'])
            except Empty:
                break
        
        with self._lock:
            self._active_connections = 0
    
    def get_stats(self) -> dict:
        """获取连接池统计信息"""
        with self._lock:
            return {
                'active_connections': self._active_connections,
                'max_connections': self.max_connections,
                'pool_size': self._pool.qsize(),
                'pool_utilization': self._active_connections / self.max_connections
            }

class BatchOperationExecutor:
    """批量操作执行器 - 重构版本"""
    
    def __init__(self, registry, cache, connectors, connection_pool_getter, logger, shared_executor=None):
        self.registry = registry
        self.cache = cache
        self.connectors = connectors
        self.get_connection_pool = connection_pool_getter
        self.logger = logger
        
        self.executor = shared_executor
        self._own_executor = shared_executor is None
        
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
    
    def execute_batch_operations(self, operations: List[BatchOperation]) -> List[BatchResult]:
        if not operations:
            return []
        
        # 按数据库ID分组操作
        operations_by_db = {}
        for op in operations:
            if op.db_id not in operations_by_db:
                operations_by_db[op.db_id] = []
            operations_by_db[op.db_id].append(op)
        
        # 提交任务到线程池
        if self.executor is None:
            raise RuntimeError("Executor is not initialized.")
        
        futures = [
            self.executor.submit(self._execute_db_operations, db_id, db_operations)
            for db_id, db_operations in operations_by_db.items()
        ]
        
        # 收集结果
        all_results = []
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                results = future.result()
                all_results.extend(results)
            except concurrent.futures.TimeoutError:
                all_results.append(BatchResult(
                    operation_id="timeout",
                    success=False,
                    error="Batch operation timeout"
                ))
            except Exception as e:
                self.logger.error(f"Batch operation error: {e}")
                all_results.append(BatchResult(
                    operation_id="error",
                    success=False,
                    error=str(e)
                ))
        
        return all_results

    def _execute_db_operations(self, db_id: str, operations: List[BatchOperation]) -> List[BatchResult]:
        """在单个数据库上执行一批操作"""
        try:
            # 使用传入的函数获取连接池
            pool = self.get_connection_pool(db_id)
            db_info = self.registry.get_database(db_id)
            connector = self.connectors[db_info.db_type]
            
            results = []
            
            # 使用单个连接执行所有操作
            with pool.get_connection() as conn:
                for op in operations:
                    try:
                        start_time = time.time()
                        result = self._execute_single_operation(conn, connector, op)
                        result.execution_time = time.time() - start_time
                        results.append(result)
                    except Exception as e:
                        results.append(BatchResult(
                            operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                            success=False,
                            error=str(e)
                        ))
            
            return results
            
        except Exception as e:
            # 连接错误，为所有操作创建错误结果
            results = []
            for op in operations:
                results.append(BatchResult(
                    operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                    success=False,
                    error=f"Connection error: {str(e)}"
                ))
            return results
    
    def _execute_single_operation(self, conn, connector, operation: BatchOperation) -> BatchResult:
        """执行单个操作"""
        operation_id = operation.operation_id or f"{operation.operation_type.value}_{hash(operation.sql)}"
                
        try:
            if operation.operation_type == OperationType.EXECUTE_QUERY:
                return self._execute_query_operation(conn, connector, operation, operation_id)
            elif operation.operation_type == OperationType.ANALYZE_PLAN:
                return self._execute_plan_operation(conn, connector, operation, operation_id)
            elif operation.operation_type == OperationType.VALIDATE_SQL:
                return self._execute_validate_operation(conn, connector, operation, operation_id)
            elif operation.operation_type == OperationType.COMPARE_SQL:
                return self._execute_compare_operation(conn, connector, operation, operation_id)
            else:
                raise ValueError(f"Unsupported operation type: {operation.operation_type}")
                
        except Exception as e:
            return BatchResult(
                operation_id=operation_id,
                success=False,
                error=str(e)
            )
    
    def _execute_query_operation(self, conn, connector, operation: BatchOperation, operation_id: str) -> BatchResult:
        """执行查询操作"""
        # 检查缓存
        cached_result = self.cache.get_query_result_smart(operation.db_id, operation.sql)
        if cached_result:
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data=cached_result
            )
        
        rows = connector.execute_query(conn, operation.sql)
        columns = self._get_columns_from_connection(conn)
        
        result_data = {
            'success': True,
            'data': rows,
            'columns': columns,
            'row_count': len(rows) if rows else 0
        }
        
        # 缓存结果
        self.cache.set_query_result(operation.db_id, operation.sql, result_data)
        
        return BatchResult(
            operation_id=operation_id,
            success=True,
            data=result_data
        )
    
    def _execute_plan_operation(self, conn, connector, operation: BatchOperation, operation_id: str) -> BatchResult:
        """执行执行计划分析操作"""
        cached_plan = self.cache.get_execution_plan(operation.db_id, operation.sql)
        if cached_plan:
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data={'execution_plan': cached_plan}
            )
        
        execution_plan = connector.get_execution_plan(conn, operation.sql)
        self.cache.set_execution_plan(operation.db_id, operation.sql, execution_plan)
        
        return BatchResult(
            operation_id=operation_id,
            success=True,
            data={'execution_plan': execution_plan}
        )
    
    def _execute_validate_operation(self, conn, connector, operation: BatchOperation, operation_id: str) -> BatchResult:
        """执行SQL验证操作"""
        try:
            connector.execute_query(conn, operation.sql)
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data={'valid': True}
            )
        except Exception:
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data={'valid': False}
            )
    
    def _execute_compare_operation(self, conn, connector, operation: BatchOperation, operation_id: str) -> BatchResult:
        """执行SQL比较操作"""
        additional_params = operation.additional_params or {}
        sql2 = additional_params.get('sql2')
        if not sql2:
            raise ValueError("sql2 required for compare operation")
        
        result1 = self._execute_query_for_comparison(conn, connector, operation.sql)
        result2 = self._execute_query_for_comparison(conn, connector, sql2)
        
        is_equal = self._compare_query_results(result1, result2)
        
        return BatchResult(
            operation_id=operation_id,
            success=True,
            data={'equal': is_equal, 'result1': result1, 'result2': result2}
        )
    
    def _get_columns_from_connection(self, conn, cursor=None) -> List[str]:
        """从连接获取列信息"""
        columns = []
        if cursor and hasattr(cursor, 'description') and cursor.description:
            columns = [desc[0] for desc in cursor.description]
        elif hasattr(conn, 'cursor'):
            temp_cursor = conn.cursor()
            try:
                if hasattr(temp_cursor, 'description') and temp_cursor.description:
                    columns = [desc[0] for desc in temp_cursor.description]
            finally:
                temp_cursor.close()
        return columns
    
    def _execute_query_for_comparison(self, conn, connector, sql: str) -> Dict:
        """为比较执行查询"""
        try:
            rows = connector.execute_query(conn, sql)
            columns = self._get_columns_from_connection(conn)
            return {
                'success': True,
                'data': rows,
                'columns': columns
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'columns': []
            }
    
    def _compare_query_results(self, result1: Dict, result2: Dict) -> bool:
        """比较查询结果"""
        if not result1['success'] or not result2['success']:
            return False
        
        data1 = result1['data']
        data2 = result2['data']
        
        if len(data1) != len(data2) or len(result1['columns']) != len(result2['columns']):
            return False
        
        def normalize_row(row):
            if isinstance(row, (list, tuple)):
                return tuple(str(item) if item is not None else None for item in row)
            return (str(row) if row is not None else None,)
        
        normalized_data1 = sorted([normalize_row(row) for row in data1])
        normalized_data2 = sorted([normalize_row(row) for row in data2])
        
        return normalized_data1 == normalized_data2

    def close(self):
        """关闭执行器"""
        if self._own_executor and hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)

class DatabaseManager:
    
    def __init__(self, db_type: str = "sqlite", config: Optional[Dict] = None, 
                 logger=None, max_connections_per_db=5, max_workers=16):
        self.db_type = db_type.lower()
        self.config = config or {}
        self.logger = logger or get_logger()
        self.max_connections_per_db = max_connections_per_db
        self.max_workers = max_workers
        
        # 核心组件
        self.registry = DatabaseRegistry(self.logger)
        self.cache = CacheManager()
        
        # 连接池管理
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.pool_lock = threading.Lock()
        
        # 统一的线程池用于并发执行
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self.connectors = {
            'sqlite': SQLiteConnector(),
            'mysql': MySQLConnector()
        }
        
        # 批量执行器 - 使用依赖注入，避免循环依赖
        self.batch_executor = BatchOperationExecutor(
            registry=self.registry,
            cache=self.cache,
            connectors=self.connectors,
            connection_pool_getter=self._get_connection_pool,  # 传递函数而不是self
            logger=self.logger,
            shared_executor=self.executor
        )
        
        # 定期清理空闲连接
        self._cleanup_timer = None
        self._cleanup_lock = threading.Lock()
        self._shutdown_requested = False
        
        # 初始化标志
        self._initialized = False
        
        try:
            self._validate_config()
            self._discover_databases()
            self._start_cleanup_timer()
            self._initialized = True
            
            if self.config.get('enable_cache_prewarming', False):
                self._prewarm_cache()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize DatabaseManager: {e}")
            self.close()
            raise
    

    def _start_cleanup_timer(self):
        """启动清理定时器"""
        with self._cleanup_lock:
            if self._cleanup_timer is None and not self._shutdown_requested:
                self._cleanup_timer = threading.Timer(300, self._periodic_cleanup)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
                self.logger.debug("Cleanup timer started")

    def _prewarm_cache(self):
        """预热缓存 - 加载常用的schema信息"""
        def prewarm_task():
            try:
                databases = self.list_databases()[:10]  # 只预热前5个数据库
                for db_id in databases:
                    try:
                        self.get_database_schema(db_id, use_cache=True)
                        self.logger.debug(f"Prewarmed cache for database: {db_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to prewarm cache for {db_id}: {e}")
            except Exception as e:
                self.logger.warning(f"Cache prewarming failed: {e}")
        
        # 异步预热，不阻塞初始化
        self.executor.submit(prewarm_task)

    def _create_error_response(self, error_msg: str = '') -> Dict[str, Any]:
        return {
            'success': False,
            'error': error_msg or 'Unknown error',
            'data': [],
            'columns': [],
            'row_count': 0
        }
    
    def _get_connection_pool(self, db_id: str) -> ConnectionPool:
        """获取或创建数据库连接池"""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
            
        if db_id not in self.connection_pools:
            with self.pool_lock:
                if db_id not in self.connection_pools:
                    db_info = self.registry.get_database(db_id)
                    if not db_info:
                        raise ValueError(f"Database {db_id} not found")
                    
                    connector = self.connectors.get(db_info.db_type)
                    if not connector:
                        raise ValueError(f"No connector found for database type: {db_info.db_type}")
                    
                    self.connection_pools[db_id] = ConnectionPool(
                        connector=connector,
                        connection_info=db_info.connection_info,
                        max_connections=self.max_connections_per_db
                    )
                    self.logger.debug(f"Created connection pool for database: {db_id}")
        
        return self.connection_pools[db_id]
    
    # ==================== 批量操作接口 ====================
    
    def batch_execute_queries(self, queries: List[Dict[str, Any]]) -> List[BatchResult]:
        """批量执行查询"""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not queries:
            return []
        
        operations = []
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            
            if not all(key in query for key in ['db_id', 'sql']):
                raise ValueError(f"Query {i} missing required fields: db_id, sql")
            
            op = BatchOperation(
                operation_type=OperationType.EXECUTE_QUERY,
                db_id=query['db_id'],
                sql=query['sql'],
                timeout=query.get('timeout', 2.0),
                operation_id=query.get('operation_id', f"query_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    def batch_analyze_execution_plans(self, queries: List[Dict[str, Any]]) -> List[BatchResult]:
        """批量分析执行计划"""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not queries:
            return []
        
        operations = []
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            
            if not all(key in query for key in ['db_id', 'sql']):
                raise ValueError(f"Query {i} missing required fields: db_id, sql")
            
            op = BatchOperation(
                operation_type=OperationType.ANALYZE_PLAN,
                db_id=query['db_id'],
                sql=query['sql'],
                timeout=query.get('timeout', 2.0),
                operation_id=query.get('operation_id', f"plan_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    def batch_validate_sql(self, queries: List[Dict[str, Any]]) -> List[BatchResult]:
        """批量验证SQL"""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not queries:
            return []
        
        operations = []
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            
            if not all(key in query for key in ['db_id', 'sql']):
                raise ValueError(f"Query {i} missing required fields: db_id, sql")
            
            op = BatchOperation(
                operation_type=OperationType.VALIDATE_SQL,
                db_id=query['db_id'],
                sql=query['sql'],
                timeout=query.get('timeout', 2.0),
                operation_id=query.get('operation_id', f"validate_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    def batch_compare_sql(self, comparisons: List[Dict[str, Any]]) -> List[BatchResult]:
        """批量比较SQL"""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not comparisons:
            return []
        
        operations = []
        for i, comp in enumerate(comparisons):
            if not isinstance(comp, dict):
                raise ValueError(f"Comparison {i} must be a dictionary")
            
            required_fields = ['db_id', 'sql1', 'sql2']
            if not all(key in comp for key in required_fields):
                raise ValueError(f"Comparison {i} missing required fields: {required_fields}")
            
            op = BatchOperation(
                operation_type=OperationType.COMPARE_SQL,
                db_id=comp['db_id'],
                sql=comp['sql1'],
                timeout=comp.get('timeout', 2.0),
                additional_params={'sql2': comp['sql2']},
                operation_id=comp.get('operation_id', f"compare_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    # ==================== 单个操作接口（统一实现）====================
    
    def _execute_single_operation_template(self, 
                                         operation_type: OperationType,
                                         db_id: str, 
                                         sql: str, 
                                         timeout: float = 2.0,
                                         additional_params: Optional[Dict] = None) -> Dict[str, Any]:
        """通用单操作执行模板"""
        if not self._initialized:
            return self._create_error_response("DatabaseManager not properly initialized")
        
        if not db_id or not sql:
            return self._create_error_response("db_id and sql are required")
        
        operation = BatchOperation(
            operation_type=operation_type,
            db_id=db_id,
            sql=sql,
            timeout=timeout,
            additional_params=additional_params
        )
        
        try:
            results = self.batch_executor.execute_batch_operations([operation])
            
            if results and results[0].success:
                return results[0].data
            else:
                error_msg = results[0].error if results and results[0].error else 'Unknown error'
                return self._create_error_response(error_msg)
        except Exception as e:
            self.logger.error(f"Error executing {operation_type.value} for {db_id}: {e}")
            return self._create_error_response(str(e))
    def execute_query(self, db_id: str, sql: str, timeout: float = 2.0) -> Dict[str, Any]:
        """执行单个查询"""
        return self._execute_single_operation_template(OperationType.EXECUTE_QUERY, db_id, sql, timeout)
    
    def analyze_sql_execution_plan(self, db_id: str, sql: str, timeout: float = 2.0) -> Dict[str, Any]:
        """分析SQL执行计划"""
        result = self._execute_single_operation_template(OperationType.ANALYZE_PLAN, db_id, sql, timeout)
        if result.get('success'):
            return {'success': True, **result}
        return {'success': False, 'error': result.get('error', 'Unknown error')}
    
    def validate_sql(self, db_id: str, sql: str, timeout: float = 2.0) -> bool:
        """验证SQL语法"""
        result = self._execute_single_operation_template(OperationType.VALIDATE_SQL, db_id, sql, timeout)
        return result.get('valid', False) if result.get('success') else False
    
    def compare_sql(self, db_id: str, sql1: str, sql2: str, timeout: float = 2.0) -> bool:
        """比较两个SQL的结果"""
        result = self._execute_single_operation_template(
            OperationType.COMPARE_SQL, db_id, sql1, timeout, {'sql2': sql2}
        )
        return result.get('equal', False) if result.get('success') else False
    
    # ==================== Schema相关方法 ====================
    
    def get_database_schema(self, db_id: str, use_cache: bool = True) -> Dict:
        """获取数据库schema（增强缓存）"""
        if not self._initialized:
            self.logger.error("DatabaseManager not properly initialized")
            return {}
        
        if not db_id:
            self.logger.error("db_id is required")
            return {}
        
        if use_cache:
            cached_schema = self.cache.get_schema(db_id, self.db_type)
            if cached_schema:
                return cached_schema
        
        db_info = self.registry.get_database(db_id)
        if not db_info:
            self.logger.error(f"Database {db_id} not found")
            return {}
        
        connector = self.connectors.get(db_info.db_type)
        if not connector:
            self.logger.error(f"No connector found for database type: {db_info.db_type}")
            return {}
        
        schema = {'tables': {}, 'foreign_keys': [], 'primary_keys': []}
        
        try:
            pool = self._get_connection_pool(db_id)
            with pool.get_connection() as conn:
                tables = connector.get_tables(conn)
                
                for table_name in tables:
                    # 检查表信息缓存
                    cached_table_info = self.cache.get_table_info(db_id, table_name)
                    if cached_table_info and use_cache:
                        table_schema = cached_table_info
                    else:
                        table_schema = connector.get_table_schema(conn, table_name)
                        sample_data = connector.get_sample_data(conn, table_name, 2)
                        
                        # 添加示例数据
                        for i, col_name in enumerate(table_schema.get('columns', {}).keys()):
                            examples = []
                            for row in sample_data.get('rows', []):
                                if i < len(row) and row[i] is not None:
                                    examples.append(str(row[i]))
                            table_schema['columns'][col_name]['examples'] = examples
                        
                        # 缓存表信息
                        if use_cache:
                            self.cache.set_table_info(db_id, table_name, table_schema)
                    
                    schema['tables'][table_name] = table_schema
                    
                    # 处理主键和外键
                    for pk in table_schema.get('primary_keys', []):
                        schema['primary_keys'].append({
                            'table': table_name,
                            'column': pk
                        })

                    for fk in table_schema.get('foreign_keys', []):
                        schema['foreign_keys'].append({
                            'source_table': table_name,
                            'source_column': fk['column'],
                            'referenced_table': fk['referenced_table'],
                            'referenced_column': fk['referenced_column']
                        })
            
            # 缓存schema
            if use_cache:
                self.cache.set_schema(db_id, self.db_type, schema)
            
        except Exception as e:
            self.logger.error(f"Error getting schema for {db_id}: {e}")
            return {}
        
        return schema

    def get_table_names_and_create_statements(self, db_id: str) -> tuple:
        """获取表名和创建语句"""
        schema = self.get_database_schema(db_id, use_cache=True)
        if not schema:
            return [], []
        
        table_names = list(schema.get('tables', {}).keys())
        create_statements = self._generate_create_statements(schema)
        return table_names, create_statements

    def get_insert_statements(self, db_id: str, table_names: Optional[List[str]] = None, limit: int = 2) -> Dict[str, List[str]]:
        """获取插入语句"""
        if not self._initialized:
            return {}
        
        if not db_id:
            return {}
        
        if limit <= 0:
            limit = 2
        
        # 统一的缓存键生成
        table_names_key = tuple(sorted(table_names or []))
        cache_key = f"insert_statements_{db_id}_{limit}_{hash(table_names_key)}"
        
        # 检查缓存
        cached_result = self.cache.get_query_result(db_id, cache_key)
        if cached_result:
            return cached_result.get('data', {})
        
        db_info = self.registry.get_database(db_id)
        if not db_info:
            return {}
        
        if table_names is None:
            schema = self.get_database_schema(db_id, use_cache=True)
            table_names = list(schema.get('tables', {}).keys())
        
        if not table_names:
            return {}
        
        connector = self.connectors.get(db_info.db_type)
        if not connector:
            return {}
        
        result = {}
        
        try:
            pool = self._get_connection_pool(db_id)
            with pool.get_connection() as conn:
                for table_name in table_names:
                    try:
                        # 表级别的缓存
                        table_cache_key = f"insert_{db_id}_{table_name}_{limit}"
                        cached_insert = self.cache.get_query_result(db_id, table_cache_key)
                        
                        if cached_insert:
                            result[table_name] = cached_insert.get('data', [])
                        else:
                            sample_data = connector.get_sample_data(conn, table_name, limit)
                            insert_statements = self._generate_insert_statements(
                                table_name, sample_data, db_info.db_type
                            )
                            result[table_name] = insert_statements
                            
                            # 缓存insert语句
                            self.cache.set_query_result(db_id, table_cache_key, {'data': insert_statements})
                    except Exception as e:
                        self.logger.warning(f"Error getting insert statements for table {table_name}: {e}")
                        result[table_name] = []
            
            # 缓存整体结果
            self.cache.set_query_result(db_id, cache_key, {'data': result})
            
        except Exception as e:
            self.logger.error(f"Error getting insert statements for {db_id}: {e}")
        
        return result
    
    def _generate_insert_statements(self, table_name: str, sample_data: Dict, db_type: str) -> List[str]:
        """生成插入语句"""
        if not sample_data.get('rows'):
            return []
        
        statements = []
        columns = sample_data.get('columns', [])
        
        if not columns:
            return []
        
        # 根据数据库类型选择引用符
        if db_type == 'mysql':
            table_quote = '`'
            col_quote = '`'
        else:
            table_quote = '"'
            col_quote = '"'
        
        for row in sample_data['rows']:
            if not row:
                continue
                
            values = []
            for value in row:
                if value is None:
                    values.append('NULL')
                elif isinstance(value, str):
                    escaped = value.replace("'", "''")
                    values.append(f"'{escaped}'")
                elif isinstance(value, (int, float)):
                    values.append(str(value))
                else:
                    values.append(f"'{str(value)}'")
            
            if values:
                columns_str = ', '.join([f'{col_quote}{col}{col_quote}' for col in columns])
                values_str = ', '.join(values)
                
                statement = f'INSERT INTO {table_quote}{table_name}{table_quote} ({columns_str}) VALUES ({values_str});'
                statements.append(statement)
        
        return statements

    # ==================== DDL 生成方法 ====================
    
    def generate_ddl_with_examples(self, db_id: str) -> str:
        """生成带示例的DDL"""
        schema = self.get_database_schema(db_id, use_cache=True)
        return self._generate_ddl(schema, use_examples=True)
    
    def generate_ddl_without_examples(self, db_id: str) -> str:
        """生成不带示例的DDL"""
        schema = self.get_database_schema(db_id, use_cache=True)
        return self._generate_ddl(schema, use_examples=False)
    
    def generate_formatted_schema_with_examples(self, db_id: str) -> str:
        """生成带示例的格式化schema"""
        schema = self.get_database_schema(db_id, use_cache=True)
        return self._generate_formatted_schema(schema, use_examples=True)
    
    def generate_formatted_schema_without_examples(self, db_id: str) -> str:
        """生成不带示例的格式化schema"""
        schema = self.get_database_schema(db_id, use_cache=True)    
        return self._generate_formatted_schema(schema, use_examples=False)
    
    def _generate_create_statements(self, schema: Dict) -> List[str]:
        """生成创建表语句"""
        statements = []
        
        for table_name, table_info in schema.get('tables', {}).items():
            columns = []
            
            for col_name, col_info in table_info.get('columns', {}).items():
                col_def = f'"{col_name}" {col_info.get("raw_type", "TEXT")}'
                if not col_info.get('nullable', True):
                    col_def += ' NOT NULL'
                if col_info.get('default'):
                    col_def += f' DEFAULT {col_info["default"]}'
                columns.append(col_def)
            
            if table_info.get('primary_keys'):
                pk_cols = ', '.join([f'"{pk}"' for pk in table_info['primary_keys']])
                columns.append(f'PRIMARY KEY ({pk_cols})')
            
            if columns:
                create_sql = f'CREATE TABLE "{table_name}" (\n    ' + ',\n    '.join(columns) + '\n);'
                statements.append(create_sql)
        
        return statements
    
    def _generate_ddl(self, schema: Dict, use_examples: bool = False) -> str:
        """生成DDL语句"""
        if not schema or not schema.get('tables'):
            return ""
        
        ddl_statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns_ddl = []
            
            for col_name, col_info in table_info.get('columns', {}).items():
                raw_type = col_info.get('raw_type', col_info.get('type', 'TEXT'))
                
                col_def = f'    "{col_name}" {raw_type}'
                
                if not col_info.get('nullable', True):
                    col_def += ' NOT NULL'
                
                if col_info.get('default'):
                    col_def += f' DEFAULT {col_info["default"]}'
                
                if use_examples and col_info.get('examples'):
                    examples = col_info['examples'][:3]
                    examples_str = ', '.join([str(ex) for ex in examples])
                    col_def += f'  -- Examples: {examples_str}'
                
                columns_ddl.append(col_def)
            
            if table_info.get('primary_keys'):
                pk_columns = ', '.join([f'"{pk}"' for pk in table_info['primary_keys']])
                columns_ddl.append(f'    PRIMARY KEY ({pk_columns})')
            
            for fk in table_info.get('foreign_keys', []):
                fk_def = (f'    FOREIGN KEY ("{fk["column"]}") '
                         f'REFERENCES "{fk["referenced_table"]}"("{fk["referenced_column"]}")')
                columns_ddl.append(fk_def)
            
            if columns_ddl:
                create_table_sql = (
                    f'CREATE TABLE "{table_name}" (\n' +
                    ',\n'.join(columns_ddl) +
                    '\n);'
                )
                ddl_statements.append(create_table_sql)
        
        # 处理全局外键
        global_fks = []
        for fk in schema.get('foreign_keys', []):
            source_table = schema['tables'].get(fk['source_table'], {})
            table_fks = source_table.get('foreign_keys', [])
            
            # 检查是否已经在表定义中包含了这个外键
            if not any(tfk.get('column') == fk['source_column'] and 
                      tfk.get('referenced_table') == fk['referenced_table'] and
                      tfk.get('referenced_column') == fk['referenced_column'] 
                      for tfk in table_fks):
                fk_sql = (f'ALTER TABLE "{fk["source_table"]}" '
                         f'ADD FOREIGN KEY ("{fk["source_column"]}") '
                         f'REFERENCES "{fk["referenced_table"]}"("{fk["referenced_column"]}");')
                global_fks.append(fk_sql)
        
        if global_fks:
            ddl_statements.extend(global_fks)
        
        return '\n\n'.join(ddl_statements)
    
    def _generate_formatted_schema(self, schema: Dict, use_examples: bool = False) -> str:
        """生成格式化的schema描述"""
        if not schema or not schema.get('tables'):
            return "No tables found in the database."
        
        formatted_parts = []
        
        table_count = len(schema['tables'])
        pk_count = len(schema.get('primary_keys', []))
        fk_count = len(schema.get('foreign_keys', []))
        
        formatted_parts.append("# Database Schema Overview")
        formatted_parts.append(f"- Total Tables: {table_count}")
        formatted_parts.append(f"- Primary Keys: {pk_count}")
        formatted_parts.append(f"- Foreign Keys: {fk_count}")
        formatted_parts.append("")
        
        for table_name, table_info in schema['tables'].items():
            formatted_parts.append(f"## Table: {table_name}")
            
            if table_info.get('primary_keys'):
                pk_list = ', '.join(table_info['primary_keys'])
                formatted_parts.append(f"**Primary Key:** {pk_list}")
            
            formatted_parts.append("**Columns:**")
            for col_name, col_info in table_info.get('columns', {}).items():
                raw_type = col_info.get('raw_type', col_info.get('type', 'TEXT'))
                
                col_desc = f"- `{col_name}` ({raw_type})"
                
                constraints = []
                if not col_info.get('nullable', True):
                    constraints.append("NOT NULL")
                if col_info.get('default'):
                    constraints.append(f"DEFAULT {col_info['default']}")
                
                if constraints:
                    col_desc += f" - {', '.join(constraints)}"
                
                if use_examples and col_info.get('examples'):
                    examples = col_info['examples'][:3]
                    examples_str = ', '.join([f"`{ex}`" for ex in examples])
                    col_desc += f" - Examples: {examples_str}"
                
                formatted_parts.append(col_desc)
            
            # 处理外键关系
            table_fks = [fk for fk in schema.get('foreign_keys', []) 
                        if fk['source_table'] == table_name]
            if table_fks:
                formatted_parts.append("**Foreign Keys:**")
                for fk in table_fks:
                    fk_desc = (f"- `{fk['source_column']}` → "
                              f"`{fk['referenced_table']}.{fk['referenced_column']}`")
                    formatted_parts.append(fk_desc)
            
            referenced_fks = [fk for fk in schema.get('foreign_keys', []) 
                             if fk['referenced_table'] == table_name]
            if referenced_fks:
                formatted_parts.append("**Referenced by:**")
                for fk in referenced_fks:
                    ref_desc = (f"- `{fk['source_table']}.{fk['source_column']}` → "
                               f"`{fk['referenced_column']}`")
                    formatted_parts.append(ref_desc)
            
            formatted_parts.append("")
        
        if schema.get('foreign_keys'):
            formatted_parts.append("## Table Relationships")
            for fk in schema['foreign_keys']:
                rel_desc = (f"- `{fk['source_table']}` → `{fk['referenced_table']}` "
                           f"({fk['source_column']} → {fk['referenced_column']})")
                formatted_parts.append(rel_desc)
        
        return '\n'.join(formatted_parts)
    

    # ==================== 数据库发现和管理方法 ====================
    
    def _validate_config(self):
        """验证配置"""
        if self.db_type not in self.connectors:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        if self.db_type == 'sqlite':
            if 'root_path' not in self.config:
                raise ValueError("SQLite requires 'root_path' in config")
            root_path = self.config['root_path']
            if not os.path.exists(root_path):
                raise ValueError(f"SQLite root path does not exist: {root_path}")
            if not os.path.isdir(root_path):
                raise ValueError(f"SQLite root path is not a directory: {root_path}")
            if not os.access(root_path, os.R_OK):
                raise ValueError(f"SQLite root path is not readable: {root_path}")
                
        elif self.db_type == 'mysql':
            required_params = ['host', 'user', 'password']
            missing = [p for p in required_params if p not in self.config]
            if missing:
                raise ValueError(f"Missing required {self.db_type} config: {missing}")
            
            # 验证端口号
            if 'port' in self.config:
                try:
                    port = int(self.config['port'])
                    if not (1 <= port <= 65535):
                        raise ValueError(f"Invalid port number: {port}")
                except (ValueError, TypeError):
                    raise ValueError(f"Port must be a valid integer: {self.config['port']}")
    
    def _discover_databases(self):
        """发现数据库"""
        self.logger.info(f"Discovering {self.db_type} databases")
        try:
            if self.db_type == 'sqlite':
                self._discover_sqlite_databases()
            elif self.db_type == 'mysql':
                self._discover_mysql_databases()
            else:
                self.logger.warning(f"Database discovery not implemented for type: {self.db_type}")
        except Exception as e:
            self.logger.error(f"Error discovering databases: {e}")
            raise
    
    def _discover_sqlite_databases(self):
        """发现SQLite数据库"""
        root_path = self.config['root_path']
        extensions = ['*.sqlite', '*.sqlite3', '*.db']
        discovered_count = 0
        
        try:
            # 首先收集所有符合条件的文件
            all_files = []
            for ext in extensions:
                pattern = os.path.join(root_path, '**', ext)
                files = glob.glob(pattern, recursive=True)
                all_files.extend(files)
            
            # 去重（可能有文件被多个模式匹配到）
            all_files = list(set(all_files))
            
            # 统一遍历所有发现的文件
            for db_file in tqdm(all_files, desc="Discovering SQLite databases"):
                if os.path.isfile(db_file):
                    try:
                        # 生成唯一的数据库ID，包含相对路径以避免冲突
                        relative_path = os.path.relpath(db_file, root_path)
                        db_id = os.path.splitext(relative_path.replace(os.sep, '_'))[0]
                            
                        # 确保ID唯一性
                        original_db_id = db_id
                        counter = 1
                        while self.registry.database_exists(db_id):
                            db_id = f"{original_db_id}_{counter}"
                            counter += 1
                            
                        # 验证文件是否为有效的SQLite数据库
                        if self._validate_sqlite_file(db_file):
                            db_info = DatabaseInfo(
                                db_id=db_id,
                                db_type='sqlite',
                                connection_info={'path': db_file},
                                metadata={
                                    'size': os.path.getsize(db_file),
                                    'modified_time': os.path.getmtime(db_file),
                                    'relative_path': relative_path
                                }
                            )
                            self.registry.register_database(db_info)
                            discovered_count += 1
                            self.logger.debug(f"Registered SQLite database: {db_id} -> {db_file}")
                        else:
                            self.logger.debug(f"Skipped invalid SQLite file: {db_file}")
                                
                    except Exception as e:
                        self.logger.warning(f"Failed to register database {db_file}: {e}")
            
            self.logger.info(f"Discovered {discovered_count} SQLite databases from {len(all_files)} files")
            
        except Exception as e:
            self.logger.error(f"Error during SQLite database discovery: {e}")
            raise
    
    def _validate_sqlite_file(self, file_path: str) -> bool:
        """验证文件是否为有效的SQLite数据库"""
        try:
            # 检查文件大小（SQLite文件至少100字节）
            if os.path.getsize(file_path) < 100:
                return False
            
            # 检查SQLite文件头
            with open(file_path, 'rb') as f:
                header = f.read(16)
                return header.startswith(b'SQLite format 3\x00')
        except Exception:
            return False
    
    def _discover_mysql_databases(self):
        """发现MySQL数据库"""
        connector = self.connectors['mysql']
        discovered_count = 0
        
        try:
            # 创建临时配置，排除database参数
            temp_config = {k: v for k, v in self.config.items() if k != 'database'}
            
            # 测试连接
            self.logger.debug("Testing MySQL connection...")
            conn = connector.connect(temp_config)
            
            try:
                # 验证连接是否正常工作
                test_result = connector.execute_query(conn, "SELECT 1 as test")
                if not test_result or test_result[0][0] != 1:
                    raise Exception("MySQL connection test failed")
                
                # 获取数据库列表
                databases = connector.execute_query(conn, "SHOW DATABASES")
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
                
                for (db_name,) in databases:
                    if db_name not in system_dbs:
                        try:
                            # 测试是否能连接到特定数据库
                            if self._test_mysql_database_access(connector, temp_config, db_name):
                                db_info = DatabaseInfo(
                                    db_id=db_name,
                                    db_type='mysql',
                                    connection_info={**self.config, 'database': db_name},
                                    metadata={
                                        'host': self.config.get('host'),
                                        'port': self.config.get('port', 3306)
                                    }
                                )
                                self.registry.register_database(db_info)
                                discovered_count += 1
                                self.logger.debug(f"Registered MySQL database: {db_name}")
                            else:
                                self.logger.debug(f"Skipped inaccessible MySQL database: {db_name}")
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to register MySQL database {db_name}: {e}")
                
                self.logger.info(f"Discovered {discovered_count} MySQL databases")
                
            finally:
                try:
                    conn.close()
                except Exception as e:
                    self.logger.warning(f"Error closing MySQL discovery connection: {e}")
                
        except Exception as e:
            self.logger.error(f"Error discovering MySQL databases: {e}")
            raise
    
    def _test_mysql_database_access(self, connector, base_config: dict, db_name: str) -> bool:
        """测试是否能访问特定的MySQL数据库"""
        try:
            test_config = {**base_config, 'database': db_name}
            test_conn = connector.connect(test_config)
            try:
                # 执行简单查询测试访问权限
                connector.execute_query(test_conn, "SELECT 1")
                return True
            finally:
                test_conn.close()
        except Exception:
            return False

    def _periodic_cleanup(self):
        """定期清理空闲连接"""
        if self._shutdown_requested:
            return
        
        try:
            self.logger.debug("Starting periodic cleanup...")
            
            with self.pool_lock:
                cleanup_count = 0
                for db_id, pool in self.connection_pools.items():
                    try:
                        initial_size = pool.get_stats().get('pool_size', 0)
                        pool.cleanup_idle_connections()
                        final_size = pool.get_stats().get('pool_size', 0)
                        
                        if initial_size > final_size:
                            cleanup_count += (initial_size - final_size)
                            self.logger.debug(f"Cleaned up {initial_size - final_size} connections for {db_id}")
                            
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up connection pool for {db_id}: {e}")
                
                if cleanup_count > 0:
                    self.logger.info(f"Periodic cleanup completed: {cleanup_count} connections cleaned")
                    
        except Exception as e:
            self.logger.error(f"Error during periodic cleanup: {e}")
        finally:
            # 重新设置定时器，但要检查是否已经关闭
            with self._cleanup_lock:
                if not self._shutdown_requested and self._cleanup_timer is not None:
                    self._cleanup_timer = threading.Timer(300, self._periodic_cleanup)
                    self._cleanup_timer.daemon = True
                    self._cleanup_timer.start()
    
    # ==================== 公共接口方法 ====================
    
    def list_databases(self) -> List[str]:
        """列出所有数据库"""
        if not self._initialized:
            self.logger.warning("DatabaseManager not initialized")
            return []
        try:
            return self.registry.list_databases()
        except Exception as e:
            self.logger.error(f"Error listing databases: {e}")
            return []
    
    def database_exists(self, db_id: str) -> bool:
        """检查数据库是否存在"""
        if not self._initialized:
            return False
        if not db_id:
            return False
        try:
            return self.registry.database_exists(db_id)
        except Exception as e:
            self.logger.error(f"Error checking database existence for {db_id}: {e}")
            return False
    
    def get_database_info(self, db_id: str) -> Optional[Dict[str, Any]]:
        """获取数据库详细信息"""
        if not self._initialized:
            return None
        
        try:
            db_info = self.registry.get_database(db_id)
            if not db_info:
                return None
            
            return {
                'db_id': db_info.db_id,
                'db_type': db_info.db_type,
                'metadata': db_info.metadata,
                'connection_info': {k: v for k, v in db_info.connection_info.items() 
                                  if k not in ['password', 'passwd']}  # 隐藏敏感信息
            }
        except Exception as e:
            self.logger.error(f"Error getting database info for {db_id}: {e}")
            return None
    
    def refresh_database_registry(self):
        """刷新数据库注册表"""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        try:
            self.logger.info("Refreshing database registry...")
            
            # 清空现有注册表和缓存
            self.registry.clear()
            self.cache.clear_all()
            
            # 清空连接池（因为数据库列表可能变化）
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    try:
                        pool.close_all()
                    except Exception as e:
                        self.logger.warning(f"Error closing pool for {db_id}: {e}")
                self.connection_pools.clear()
            
            # 重新发现数据库
            self._discover_databases()
            
            self.logger.info("Database registry refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Error refreshing database registry: {e}")
            raise
    
    def clear_cache(self):
        """清空缓存"""
        try:
            self.cache.clear_all()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            raise

    def close(self):
        """关闭所有连接和资源"""
        if self._shutdown_requested:
            self.logger.debug("DatabaseManager already shutting down")
            return
        
        self._shutdown_requested = True
        self.logger.info("Shutting down DatabaseManager...")
        
        try:
            # 1. 停止清理定时器
            with self._cleanup_lock:
                if self._cleanup_timer is not None:
                    try:
                        self._cleanup_timer.cancel()
                        self.logger.debug("Cleanup timer cancelled")
                    except Exception as e:
                        self.logger.warning(f"Error cancelling cleanup timer: {e}")
                    finally:
                        self._cleanup_timer = None
            
            # 2. 关闭批量执行器（如果有独立的线程池）
            if hasattr(self.batch_executor, 'close'):
                try:
                    self.batch_executor.close()
                    self.logger.debug("Batch executor closed")
                except Exception as e:
                    self.logger.warning(f"Error closing batch executor: {e}")
            
            # 3. 关闭所有连接池
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    try:
                        pool.close_all()
                        self.logger.debug(f"Closed connection pool for {db_id}")
                    except Exception as e:
                        self.logger.warning(f"Error closing connection pool for {db_id}: {e}")
                self.connection_pools.clear()
            
            # 4. 关闭线程池 - 修正语法错误
            if hasattr(self, 'executor') and self.executor:
                try:
                    # 先尝试优雅关闭
                    self.executor.shutdown(wait=False)
                    
                    # 等待最多10秒
                    start_time = time.time()
                    while not self.executor._shutdown and time.time() - start_time < 10:
                        time.sleep(0.1)
                    
                    # 如果还没关闭，强制关闭
                    if not self.executor._shutdown:
                        self.logger.warning("Thread pool did not shut down gracefully, forcing shutdown")
                        self.executor.shutdown(wait=True)
                    
                    self.logger.debug("Thread pool shut down successfully")
                    
                except Exception as e:
                    self.logger.warning(f"Error shutting down thread pool: {e}")
            
            # 5. 清空缓存
            try:
                self.cache.clear_all()
                self.logger.debug("Cache cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing cache: {e}")
            
            # 6. 清空注册表
            try:
                self.registry.clear()
                self.logger.debug("Registry cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing registry: {e}")
            
            # 7. 更新状态
            self._initialized = False
            self.logger.info("DatabaseManager shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self._initialized:
            return {'error': 'DatabaseManager not initialized', 'initialized': False}
        
        try:
            # 获取缓存统计
            cache_stats = self.cache.get_cache_stats()
            
            # 连接池统计
            pool_stats = {}
            total_connections = 0
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    try:
                        stats = pool.get_stats()
                        pool_stats[db_id] = stats
                        total_connections += stats.get('active_connections', 0)
                    except Exception as e:
                        pool_stats[db_id] = {'error': str(e)}
            
            # 线程池统计
            thread_pool_stats = {
                'max_workers': self.max_workers,
                'active_threads': getattr(self.executor, '_threads', 0) if hasattr(self.executor, '_threads') else 'unknown'
            }
            
            # 数据库统计
            database_count = len(self.list_databases())
            
            return {
                'initialized': self._initialized,
                'shutdown_requested': self._shutdown_requested,
                'database_count': database_count,
                'db_type': self.db_type,
                'max_workers': self.max_workers,
                'max_connections_per_db': self.max_connections_per_db,
                'total_active_connections': total_connections,
                'cache_stats': cache_stats,
                'connection_pools': pool_stats,
                'active_pools': len(self.connection_pools),
                'thread_pool_stats': thread_pool_stats,
                'cleanup_timer_active': self._cleanup_timer is not None and self._cleanup_timer.is_alive()
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'error': str(e), 'initialized': self._initialized}
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        try:
            # 检查初始化状态
            health_status['checks']['initialization'] = {
                'status': 'pass' if self._initialized else 'fail',
                'message': 'System initialized' if self._initialized else 'System not initialized'
            }
            
            # 检查数据库注册表
            try:
                db_count = len(self.list_databases())
                health_status['checks']['database_registry'] = {
                    'status': 'pass',
                    'message': f'{db_count} databases registered',
                    'count': db_count
                }
            except Exception as e:
                health_status['checks']['database_registry'] = {
                    'status': 'fail',
                    'message': f'Registry error: {str(e)}'
                }
            
            # 检查线程池
            try:
                if hasattr(self.executor, '_shutdown') and self.executor._shutdown:
                    health_status['checks']['thread_pool'] = {
                        'status': 'fail',
                        'message': 'Thread pool is shut down'
                    }
                else:
                    health_status['checks']['thread_pool'] = {
                        'status': 'pass',
                        'message': f'Thread pool operational (max workers: {self.max_workers})'
                    }
            except Exception as e:
                health_status['checks']['thread_pool'] = {
                    'status': 'fail',
                    'message': f'Thread pool error: {str(e)}'
                }
            
            # 检查连接池
            try:
                with self.pool_lock:
                    pool_count = len(self.connection_pools)
                    health_status['checks']['connection_pools'] = {
                        'status': 'pass',
                        'message': f'{pool_count} connection pools active',
                        'count': pool_count
                    }
            except Exception as e:
                health_status['checks']['connection_pools'] = {
                    'status': 'fail',
                    'message': f'Connection pool error: {str(e)}'
                }
            
            # 检查缓存
            try:
                cache_stats = self.cache.get_cache_stats()
                total_size = cache_stats.get('total_size', 0)
                health_status['checks']['cache'] = {
                    'status': 'pass',
                    'message': f'Cache operational (size: {total_size})',
                    'size': total_size
                }
            except Exception as e:
                health_status['checks']['cache'] = {
                    'status': 'fail',
                    'message': f'Cache error: {str(e)}'
                }
            
            # 检查清理定时器
            timer_status = (self._cleanup_timer is not None and 
                          self._cleanup_timer.is_alive() and 
                          not self._shutdown_requested)
            
            health_status['checks']['cleanup_timer'] = {
                'status': 'pass' if timer_status else 'warn',
                'message': 'Cleanup timer running' if timer_status else 'Cleanup timer not running'
            }
            
            # 综合状态评估
            failed_checks = [name for name, check in health_status['checks'].items() 
                           if check['status'] == 'fail']
            warning_checks = [name for name, check in health_status['checks'].items() 
                            if check['status'] == 'warn']
            
            if failed_checks:
                health_status['status'] = 'unhealthy'
                health_status['failed_checks'] = failed_checks
            elif warning_checks:
                health_status['status'] = 'degraded'
                health_status['warning_checks'] = warning_checks
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    

    
    def __enter__(self):
        """上下文管理器进入"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, '_initialized') and self._initialized and not self._shutdown_requested:
                self.close()
        except Exception:
            pass  # 忽略析构函数中的异常