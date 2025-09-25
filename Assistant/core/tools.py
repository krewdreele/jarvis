import inspect
from types import UnionType
from typing import Any, Callable, Dict, Literal, Union, get_args, get_origin, get_type_hints

from openai.types.responses import FunctionToolParam


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    return (origin is UnionType or origin is Union) and type(None) in args


def _strip_optional(annotation: Any) -> Any:
    if not _is_optional(annotation):
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    raise TypeError(f"Unsupported Union with multiple non-None values: {annotation}")


def _get_strict_json_schema_type(annotation: Any) -> Dict[str, Any]:
    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)

    type_map: Dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    if annotation in type_map:
        return {"type": type_map[annotation]}

    if origin in type_map:
        return {"type": type_map[origin]}

    if origin is Literal:
        values = args
        if all(isinstance(v, (str, int, bool)) for v in values):
            literal_type = "string"
            if all(isinstance(v, bool) for v in values):
                literal_type = "boolean"
            elif all(isinstance(v, int) for v in values):
                literal_type = "integer"
            return {"type": literal_type, "enum": list(values)}
        raise TypeError("Unsupported Literal values in annotation")

    raise TypeError(f"Unsupported parameter type: {annotation}")


def generate_function_schema(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionToolParam:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    params: Dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in {"self", "ctx"}:
            continue

        ann = type_hints.get(param_name, param.annotation)
        if ann is inspect._empty:
            raise TypeError(f"Missing type annotation for parameter: {param_name}")

        schema_entry = _get_strict_json_schema_type(ann)
        has_default = param.default is not inspect._empty
        if has_default and param.default is not None:
            schema_entry["default"] = param.default

        is_required = not has_default and not _is_optional(ann)
        if is_required:
            required.append(param_name)

        params[param_name] = schema_entry

    return {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": description or (func.__doc__ or ""),
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
                "additionalProperties": False,
            },
        },
        "strict": True,
    }


class ToolBox:
    tools: list[FunctionToolParam]

    def __init__(self, namespace: str | None = None) -> None:
        self._namespace = namespace
        self._funcs: Dict[str, Callable[..., Any]] = {}
        self.tools = []

    def _qualify(self, name: str) -> str:
        if self._namespace and not name.startswith(f"{self._namespace}_"):
            return f"{self._namespace}_{name}"
        return name

    def qualify(self, name: str) -> str:
        """Return the fully-qualified tool name for a local function name."""
        return self._qualify(name)

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = self._qualify(name or inner.__name__)
            schema = generate_function_schema(inner, name=tool_name, description=description)
            self._funcs[tool_name] = inner
            self._upsert_tool_schema(schema)
            return inner

        if func is None:
            return decorator
        return decorator(func)

    def _tool_name(self, schema: FunctionToolParam) -> str:
        fn = schema.get("function") or {}
        return fn.get("name") or ""

    def _upsert_tool_schema(self, schema: FunctionToolParam) -> None:
        target_name = self._tool_name(schema)
        self.tools = [t for t in self.tools if self._tool_name(t) != target_name]
        self.tools.append(schema)

    def get_tool_function(self, tool_name: str) -> Callable[..., Any] | None:
        return self._funcs.get(tool_name)

    def handlers(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._funcs)

    def extend(self, other: "ToolBox") -> None:
        for schema in other.tools:
            self._upsert_tool_schema(schema)
        self._funcs.update(other._funcs)

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        func = self.get_tool_function(tool_name)
        if func is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        return func(**arguments)
