"""
代码执行环境

安全的Python代码执行环境，用于评估Agent生成的代码
"""
import re
import ast
import sys
import io
import traceback
from typing import Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr


class CodeExecutionEnv:
    """
    安全的代码执行环境
    
    功能：
    1. 提取Agent响应中的代码
    2. 在沙箱中执行
    3. 返回执行结果和奖励
    """
    
    def __init__(
        self,
        timeout: float = 5.0,
        max_output_length: int = 1000,
        allow_imports: Optional[list] = None,
    ):
        self.timeout = timeout
        self.max_output_length = max_output_length
        
        # 允许的导入（白名单）
        if allow_imports is None:
            allow_imports = [
                'math', 'random', 'datetime', 'itertools', 'collections',
                'statistics', 'functools', 'decimal', 'fractions'
            ]
        self.allow_imports = set(allow_imports)
    
    def extract_code(self, text: str) -> Optional[str]:
        """
        从文本中提取代码
        
        支持的格式：
        - ```python\ncode\n```
        - ```\ncode\n```
        - 直接代码（如果没有markdown代码块）
        """
        # 尝试提取python代码块
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 尝试提取无标签代码块
        pattern = r'```\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没有代码块，尝试直接解析
        # 检查是否是有效的Python代码
        lines = text.strip().split('\n')
        code_lines = []
        for line in lines:
            # 跳过空行和明显不是代码的行
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return None
    
    def validate_code(self, code: str) -> tuple:
        """
        验证代码安全性
        
        Returns:
            (is_safe, error_message)
        """
        try:
            # 解析AST
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # 检查危险的导入和调用
        for node in ast.walk(tree):
            # 检查导入
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allow_imports:
                        return False, f"Import '{alias.name}' not allowed"
            
            if isinstance(node, ast.ImportFrom):
                if node.module not in self.allow_imports:
                    return False, f"Import from '{node.module}' not allowed"
            
            # 检查危险函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                        return False, f"Function '{node.func.id}' not allowed"
                
                if isinstance(node.func, ast.Attribute):
                    # 检查open()等文件操作
                    if node.func.attr in ['open', 'read', 'write']:
                        return False, f"File operation '{node.func.attr}' not allowed"
        
        return True, ""
    
    def execute(
        self,
        code: str,
        test_input: Optional[Any] = None,
        expected_output: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        执行代码
        
        Args:
            code: Python代码
            test_input: 测试输入（可选）
            expected_output: 期望输出（可选）
            
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "result": Any,
                "execution_time": float,
            }
        """
        import time
        
        result = {
            "success": False,
            "output": "",
            "error": "",
            "result": None,
            "execution_time": 0.0,
        }
        
        # 验证代码
        is_safe, error_msg = self.validate_code(code)
        if not is_safe:
            result["error"] = error_msg
            return result
        
        # 准备执行环境
        namespace = {
            '__builtins__': {
                name: __builtins__[name]
                for name in ['abs', 'all', 'any', 'bin', 'bool', 'chr', 'divmod',
                           'enumerate', 'filter', 'float', 'format', 'frozenset',
                           'hasattr', 'hash', 'hex', 'int', 'isinstance', 'issubclass',
                           'iter', 'len', 'list', 'map', 'max', 'min', 'next',
                           'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed',
                           'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple',
                           'type', 'zip', 'True', 'False', 'None']
            }
        }
        
        # 添加允许的模块
        for module_name in self.allow_imports:
            try:
                module = __import__(module_name)
                namespace[module_name] = module
            except ImportError:
                pass
        
        # 添加测试输入
        if test_input is not None:
            namespace['input'] = test_input
        
        # 执行代码
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        start_time = time.time()
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, namespace)
            
            execution_time = time.time() - start_time
            
            # 获取结果
            output = stdout_buffer.getvalue()
            error = stderr_buffer.getvalue()
            
            # 截断输出
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "... [truncated]"
            
            result["output"] = output
            result["error"] = error
            result["execution_time"] = execution_time
            result["success"] = True
            
            # 如果有期望输出，进行对比
            if expected_output is not None:
                # 尝试从namespace获取结果
                if 'result' in namespace:
                    actual_result = namespace['result']
                    result["result"] = actual_result
                    
                    if actual_result == expected_output:
                        result["correct"] = True
                    else:
                        result["correct"] = False
                        result["error"] = f"Expected {expected_output}, got {actual_result}"
            
        except Exception as e:
            result["error"] = traceback.format_exc()
            result["execution_time"] = time.time() - start_time
        
        return result
    
    def evaluate_response(
        self,
        prompt: str,
        response: str,
        test_cases: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        评估Agent的响应
        
        Args:
            prompt: 任务描述
            response: Agent的响应
            test_cases: 测试用例列表 [{"input": ..., "expected": ...}]
            
        Returns:
            评估结果和奖励
        """
        # 提取代码
        code = self.extract_code(response)
        
        if code is None:
            return {
                "success": False,
                "error": "No code found in response",
                "reward": -1.0,
            }
        
        # 执行代码
        if test_cases:
            # 使用测试用例评估
            passed = 0
            total = len(test_cases)
            
            for test in test_cases:
                result = self.execute(
                    code,
                    test_input=test.get("input"),
                    expected_output=test.get("expected"),
                )
                
                if result.get("correct", False):
                    passed += 1
            
            accuracy = passed / total if total > 0 else 0
            
            return {
                "success": passed == total,
                "code": code,
                "accuracy": accuracy,
                "passed": passed,
                "total": total,
                "reward": accuracy * 2 - 1,  # 归一化到[-1, 1]
            }
        else:
            # 没有测试用例，只检查代码能否运行
            result = self.execute(code)
            
            return {
                "success": result["success"],
                "code": code,
                "output": result["output"],
                "error": result["error"],
                "reward": 1.0 if result["success"] else -1.0,
            }
