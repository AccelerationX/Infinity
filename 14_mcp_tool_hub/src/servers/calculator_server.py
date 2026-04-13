"""
Calculator MCP Server（FastMCP 标准实现）
提供安全的数学表达式计算
"""
import ast
import operator as op
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

# 允许的操作符映射
_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}


def _eval_node(node):
    """递归求值 AST 节点，拒绝任何非纯数学结构"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
        return _OPERATORS[op_type](operand)
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


@mcp.tool()
async def calculate(expression: str) -> str:
    """
    安全计算数学表达式。
    支持：+ - * / // % ** 和括号。禁止函数调用、变量访问等。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        return str(result)
    except Exception as e:
        return f"Error: Invalid expression ({e})"


if __name__ == "__main__":
    mcp.run(transport="stdio")
