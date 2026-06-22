"""
ComfyUI custom node pack: OpenAI-compatible LLM caller

Install:
  1. Put this file into: ComfyUI/custom_nodes/comfyui_openai_compatible_llm_node.py
  2. Optional: pip install python-dotenv
  3. Put OPENAI_API_KEY=... in your environment or .env file when using OpenAI/OpenRouter/etc.
     Local servers such as LM Studio/Ollama can leave the key empty.
  4. Restart ComfyUI.

Examples:
  OpenAI:    api_base_url = https://api.openai.com/v1
  LM Studio: api_base_url = http://127.0.0.1:1234/v1
  Ollama:    api_base_url = http://127.0.0.1:11434/v1

This node intentionally uses only the Python standard library for HTTP calls.
python-dotenv is optional; if installed, load_dotenv() is called.

Optional unload_after_call supports provider-specific cleanup for LM Studio and Ollama.
Unload uses the same Authorization header as the main request.
Unload failures are returned as warnings in unload_json; the main LLM output is preserved.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import signal
import subprocess
import threading
import queue
import time
import atexit
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# dotenv support
# -----------------------------------------------------------------------------

def _fallback_load_dotenv(filename: str = ".env") -> None:
    """Tiny fallback .env reader used only when python-dotenv is not installed."""
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            # Do not fail node import because .env parsing failed.
            pass


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        _fallback_load_dotenv()


_load_dotenv_if_available()


# JavaScript extension directory for the dynamic MCP Tools Stack input UI.
WEB_DIRECTORY = "./js"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

class HttpJsonError(RuntimeError):
    """HTTP error that preserves status code and response body for safe fallback."""

    def __init__(self, status_code: int, body: str, url: str):
        self.status_code = int(status_code)
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status_code} from LLM server: {body}")


class ContainsAnyDict(dict):
    """Dict that lets ComfyUI backend accept dynamically-created MCP_TOOL input names.

    ComfyUI validation may both check `name in optional` and then index
    `optional[name]`. Returning True from __contains__ is not enough: without
    __getitem__, dynamically-created inputs such as `tool_1` raise KeyError
    during prompt validation.
    """

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: object):
        return ("MCP_TOOL", {"forceInput": True})

    def get(self, key: object, default=None):
        return self[key]


def _join_url(base: str, suffix: str) -> str:
    return base.rstrip("/") + suffix


def _chat_completions_sibling_url(endpoint_url: str) -> str:
    """Return the sibling /chat/completions endpoint for a /responses URL."""
    url = (endpoint_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/responses"):
        new_path = path[: -len("/responses")] + "/chat/completions"
        return urllib.parse.urlunparse(parsed._replace(path=new_path))
    return _resolve_api_endpoint(url)[1]


def _resolve_api_endpoint(api_base_url: str) -> Tuple[str, str, Optional[str]]:
    """
    Resolve api_base_url into (mode, endpoint, fallback_endpoint).

    Modes:
      chat       -> send Chat Completions payload to /chat/completions
      responses  -> send Responses payload to /responses, with chat fallback on endpoint-not-supported errors

    Supported URL forms:
      .../v1                    -> chat, appends /chat/completions
      .../v1/chat/completions   -> chat
      .../v1/responses          -> responses, fallback sibling is .../v1/chat/completions
    """
    url = (api_base_url or "").strip()
    if not url:
        raise ValueError("api_base_url is empty")

    url = url.rstrip("/")
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")

    if path.endswith("/responses"):
        return "responses", url, _chat_completions_sibling_url(url)

    if path.endswith("/chat/completions"):
        return "chat", url, None

    if path == "" or path == "/":
        return "chat", _join_url(url, "/v1/chat/completions"), None

    if path.endswith("/v1"):
        return "chat", _join_url(url, "/chat/completions"), None

    # Preserve the old forgiving behavior for custom base paths.
    return "chat", _join_url(url, "/chat/completions"), None


def _is_responses_endpoint_not_supported_error(error: Exception) -> bool:
    """Only fallback when the endpoint itself appears unsupported, not for auth/model/schema errors."""
    if not isinstance(error, HttpJsonError):
        return False
    if error.status_code in (404, 405, 501):
        return True
    body = (error.body or "").lower()
    return error.status_code == 400 and (
        "responses" in body and any(token in body for token in ("not found", "unsupported", "unknown endpoint", "unknown route"))
    )


def _parse_json_object(value: str, field_name: str) -> Dict[str, Any]:
    text = (value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{field_name} must be a JSON object: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed




def _parse_json_array(value: str, field_name: str) -> List[Any]:
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{field_name} must be a JSON array: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return parsed


def _parse_string_list(value: str, field_name: str = "string list") -> List[str]:
    """Parse a JSON string array, comma-separated list, or newline-separated list."""
    text = (value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = _parse_json_array(text, field_name)
        if not all(isinstance(x, str) for x in parsed):
            raise ValueError(f"{field_name} must contain only strings")
        return [x.strip() for x in parsed if x.strip()]

    # Support both newline and comma separators for quick ComfyUI editing.
    items: List[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items



def _normalize_mcp_tools(mcp_tools: Any) -> List[Dict[str, Any]]:
    """
    Normalize an MCP_TOOLS/MCP_TOOL/JSON value into a list of LLM tool dictionaries.

    Backward compatibility note: this function name says mcp_tools because older
    workflows use MCP_TOOL/MCP_TOOLS sockets. In v12 it accepts both:
      - Remote MCP tools: {"type":"mcp", ...} for Responses API passthrough.
      - Local MCP commands: {"kind":"local_mcp", ...} executed by this node.
    """
    if mcp_tools is None:
        return []

    parsed: Any = mcp_tools
    if isinstance(mcp_tools, str):
        text = mcp_tools.strip()
        if not text:
            return []
        parsed = json.loads(text)

    if isinstance(parsed, dict):
        # Accept either a single tool object or {"tools": [...]} for convenience.
        if isinstance(parsed.get("tools"), list):
            parsed = parsed["tools"]
        else:
            parsed = [parsed]

    if isinstance(parsed, tuple):
        parsed = list(parsed)

    if not isinstance(parsed, list):
        raise ValueError("mcp_tools must be a tool, tools list, JSON object, or JSON array")

    tools: List[Dict[str, Any]] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"mcp_tools[{i}] must be a JSON object")
        kind = _llm_tool_kind(item)
        if kind not in ("remote_mcp", "local_mcp"):
            raise ValueError(
                f"mcp_tools[{i}] must be a Remote MCP tool with type='mcp' "
                f"or a Local MCP tool with kind='local_mcp'"
            )
        tools.append(dict(item))
    return tools


def _llm_tool_kind(tool: Dict[str, Any]) -> str:
    if tool.get("kind") == "local_mcp" or tool.get("type") == "local_mcp":
        return "local_mcp"
    if tool.get("kind") == "remote_mcp" or tool.get("type") == "mcp":
        return "remote_mcp"
    return "unknown"


def _split_llm_tools(mcp_tools: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    remote: List[Dict[str, Any]] = []
    local: List[Dict[str, Any]] = []
    for tool in _normalize_mcp_tools(mcp_tools):
        if _llm_tool_kind(tool) == "local_mcp":
            local.append(tool)
        else:
            # Strip internal kind if present; Responses API expects type=mcp.
            t = dict(tool)
            t.pop("kind", None)
            t["type"] = "mcp"
            remote.append(t)
    return remote, local


def _merge_mcp_tools_into_body(body: Dict[str, Any], mcp_tools: Any) -> None:
    remote_tools, _local_tools = _split_llm_tools(mcp_tools)
    if not remote_tools:
        return
    existing = body.get("tools")
    if existing is None:
        body["tools"] = remote_tools
    elif isinstance(existing, list):
        body["tools"] = existing + remote_tools
    else:
        raise ValueError("extra_body_json.tools must be a JSON array when mcp_tools is also connected")


_ENV_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _expand_env_placeholders(text: str, field_name: str) -> str:
    """Expand {{ENV_NAME}} placeholders in a string.

    Bare values are intentionally not treated as env vars. This avoids making
    ordinary literals such as lang=ja or transport=sse ambiguous.
    """
    value = "" if text is None else str(text)

    def repl(match: re.Match[str]) -> str:
        env_name = match.group(1)
        env_value = os.environ.get(env_name, "")
        if not env_value:
            raise ValueError(f"{field_name} references environment variable {env_name!r}, but it is empty")
        return env_value

    return _ENV_PLACEHOLDER_RE.sub(repl, value)


def _expand_env_placeholders_in_url(url: str) -> str:
    """Expand {{ENV_NAME}} placeholders in server_url.

    This is intentionally supported as a flexible URL templating feature. Query
    parameters are parsed and re-encoded so secrets or values containing special
    characters are encoded safely.
    """
    if "{{" not in (url or ""):
        return url

    parts = urllib.parse.urlsplit(url)

    query_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    if query_pairs:
        query = urllib.parse.urlencode(
            [
                (
                    _expand_env_placeholders(key, "server_url query parameter name"),
                    _expand_env_placeholders(value, f"server_url query parameter {key!r}"),
                )
                for key, value in query_pairs
            ]
        )
    else:
        query = ""

    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            _expand_env_placeholders(parts.path, "server_url path"),
            query,
            _expand_env_placeholders(parts.fragment, "server_url fragment"),
        )
    )


def _expand_env_placeholders_in_json_value(value: Any, field_name: str) -> Any:
    """Recursively expand {{ENV_NAME}} placeholders inside parsed JSON values."""
    if isinstance(value, str):
        return _expand_env_placeholders(value, field_name)
    if isinstance(value, list):
        return [
            _expand_env_placeholders_in_json_value(item, f"{field_name}[{i}]")
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            str(key): _expand_env_placeholders_in_json_value(val, f"{field_name}.{key}")
            for key, val in value.items()
        }
    return value


def _stringify_query_param_value(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} contains null, which cannot be used as a query parameter value")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError(f"{field_name} values must be strings, numbers, or booleans")


def _parse_query_params_json(value: str, field_name: str) -> Dict[str, str]:
    text = (value or "").strip()
    if not text:
        return {}

    parsed = _parse_json_object(text, field_name)
    out: Dict[str, str] = {}

    for key, param_value in parsed.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError(f"{field_name} contains an empty query parameter name")

        raw_value = _stringify_query_param_value(param_value, field_name)
        out[key_text] = _expand_env_placeholders(raw_value, f"{field_name}.{key_text}")

    return out


def _append_query_params_to_url(url: str, params: Dict[str, str]) -> str:
    if not params:
        return url

    parts = urllib.parse.urlsplit(url)
    existing = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)

    # Last write wins for duplicate names. This makes explicit query_params_json
    # values override parameters already present in server_url.
    ordered_names: List[str] = []
    merged: Dict[str, str] = {}

    for key, value in existing:
        if key not in merged:
            ordered_names.append(key)
        merged[key] = value

    for key, value in params.items():
        if key not in merged:
            ordered_names.append(key)
        merged[key] = value

    query = urllib.parse.urlencode([(key, merged[key]) for key in ordered_names])
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _build_remote_mcp_tool(
    server_label: str,
    server_url: str,
    allowed_tools: str = "",
    server_description: str = "",
    headers_json: str = "",
    authorization_env: str = "",
    query_params_json: str = "",
) -> Dict[str, Any]:
    label = (server_label or "").strip()
    url = (server_url or "").strip()

    if not label:
        raise ValueError("server_label is required")
    if not url:
        raise ValueError("server_url is required")

    url = _expand_env_placeholders_in_url(url)
    url = _append_query_params_to_url(url, _parse_query_params_json(query_params_json, "query_params_json"))

    tool: Dict[str, Any] = {
        "type": "mcp",
        "server_label": label,
        "server_url": url,
        # ComfyUI has no interactive Remote MCP approval UI, so Remote MCP
        # approval is intentionally fixed to "never".
        "require_approval": "never",
    }

    if (server_description or "").strip():
        tool["server_description"] = server_description.strip()

    allowed = _parse_string_list(allowed_tools, "allowed_tools")
    if allowed:
        tool["allowed_tools"] = allowed

    headers = _parse_json_object(headers_json, "headers_json") if (headers_json or "").strip() else {}
    if headers:
        tool["headers"] = _expand_env_placeholders_in_json_value(headers, "headers_json")

    auth_env = (authorization_env or "").strip()
    if auth_env:
        token = os.environ.get(auth_env, "")
        if not token:
            raise ValueError(f"authorization_env is set to {auth_env!r}, but that environment variable is empty")
        tool["authorization"] = token

    return tool



# -----------------------------------------------------------------------------
# Local stdio MCP runtime
# -----------------------------------------------------------------------------

_ACTIVE_MCP_SESSIONS: List["_LocalMCPSession"] = []
_ACTIVE_MCP_LOCK = threading.Lock()


def _cleanup_active_mcp_sessions() -> None:
    with _ACTIVE_MCP_LOCK:
        sessions = list(_ACTIVE_MCP_SESSIONS)
    for session in sessions:
        try:
            session.close()
        except Exception:
            pass


atexit.register(_cleanup_active_mcp_sessions)


def _safe_tool_name(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip())
    value = value.strip("_") or "tool"
    if not re.match(r"^[a-zA-Z_]", value):
        value = "tool_" + value
    return value[:64]


def _json_rpc_error_to_text(error: Any) -> str:
    if isinstance(error, dict):
        return json.dumps(error, ensure_ascii=False)
    return str(error)


class _LocalMCPSession:
    """Minimal stdio MCP client used only for one ComfyUI node execution."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.server_label = _safe_tool_name(str(spec.get("server_label") or "local_mcp"))
        self.command = str(spec.get("command") or "").strip()
        self.args = list(spec.get("args") or [])
        self.env_extra = dict(spec.get("env") or {})
        self.startup_timeout_sec = int(spec.get("startup_timeout_sec") or 15)
        self.tool_timeout_sec = int(spec.get("tool_timeout_sec") or 60)
        self.process: Optional[subprocess.Popen[str]] = None
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._stderr_lines: List[str] = []
        self._next_id = 1
        self._closed = False

    def start(self) -> None:
        if not self.command:
            raise ValueError("Local MCP command is required")
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.env_extra.items()})

        kwargs: Dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        self.process = subprocess.Popen([self.command] + [str(a) for a in self.args], **kwargs)
        with _ACTIVE_MCP_LOCK:
            _ACTIVE_MCP_SESSIONS.append(self)

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        # MCP initialize handshake.
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "workflow-knives-comfyui", "version": "0.12"},
            },
            timeout=self.startup_timeout_sec,
        )
        self.notify("notifications/initialized", {})

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except Exception:
                # stdout should be JSON-RPC only, but keep non-JSON lines for debugging.
                self._queue.put({"__non_json_stdout__": line})

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            if line:
                self._stderr_lines.append(line.rstrip())
                if len(self._stderr_lines) > 200:
                    del self._stderr_lines[:50]

    def _send(self, payload: Dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP process is not running")
        if self.process.poll() is not None:
            raise RuntimeError(f"MCP process exited with code {self.process.returncode}. stderr: {self.stderr_tail()}")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Any:
        request_id = self._next_id
        self._next_id += 1
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

        deadline = time.time() + float(timeout or self.tool_timeout_sec)
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(f"MCP process exited with code {self.process.returncode}. stderr: {self.stderr_tail()}")
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("id") != request_id:
                # Notification or unrelated response. Ignore for this minimal runner.
                continue
            if "error" in msg:
                raise RuntimeError(f"MCP {method} error: {_json_rpc_error_to_text(msg.get('error'))}")
            return msg.get("result")
        raise TimeoutError(f"MCP request timed out: {method}. stderr: {self.stderr_tail()}")

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self.request("tools/list", {}, timeout=self.startup_timeout_sec)
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=self.tool_timeout_sec)

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines[-20:])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self.process
        with _ACTIVE_MCP_LOCK:
            try:
                _ACTIVE_MCP_SESSIONS.remove(self)
            except ValueError:
                pass
        if proc is None:
            return
        try:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                if os.name == "nt":
                    # Terminate process tree on Windows. This is important for npx wrappers.
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=5,
                        )
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    if proc.poll() is None:
                        if os.name == "nt":
                            try:
                                subprocess.run(
                                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    timeout=5,
                                )
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                        else:
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
        finally:
            self.process = None


class _LocalMCPToolRuntime:
    def __init__(self, local_specs: List[Dict[str, Any]]):
        self.local_specs = local_specs
        self.sessions: List[_LocalMCPSession] = []
        self.tool_map: Dict[str, Tuple[_LocalMCPSession, str]] = {}
        self.trace: List[Dict[str, Any]] = []

    def __enter__(self) -> "_LocalMCPToolRuntime":
        for spec in self.local_specs:
            session = _LocalMCPSession(spec)
            session.start()
            self.sessions.append(session)
            allowed = set(str(x) for x in (spec.get("allowed_tools") or []) if str(x).strip())
            for tool in session.list_tools():
                original_name = str(tool.get("name") or "").strip()
                if not original_name:
                    continue
                exposed_name = _safe_tool_name(f"{session.server_label}__{original_name}")
                if allowed and original_name not in allowed and exposed_name not in allowed:
                    continue
                self.tool_map[exposed_name] = (session, original_name)
                self.trace.append({"event": "local_mcp_tool_listed", "server_label": session.server_label, "tool": original_name, "exposed_name": exposed_name})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for session in reversed(self.sessions):
            try:
                session.close()
            except Exception as e:
                self.trace.append({"event": "local_mcp_close_error", "server_label": session.server_label, "error": str(e)})
        self.sessions.clear()

    def function_tools_for_responses(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for exposed_name, (session, original_name) in self.tool_map.items():
            # Find schema again from trace/session cache is not retained, so use a compact default unless cached below.
            pass
        return tools

    def build_function_tools(self, api_mode: str) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        # Build from listed trace entries and fresh tool metadata cached in _tool_meta.
        for exposed_name, meta in getattr(self, "_tool_meta", {}).items():
            description = str(meta.get("description") or f"Local MCP tool {exposed_name}")
            parameters = meta.get("inputSchema") or meta.get("input_schema") or {"type": "object", "properties": {}}
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            if api_mode == "responses":
                tools.append({"type": "function", "name": exposed_name, "description": description, "parameters": parameters})
            else:
                tools.append({"type": "function", "function": {"name": exposed_name, "description": description, "parameters": parameters}})
        return tools

    def cache_tool_meta(self) -> None:
        meta: Dict[str, Dict[str, Any]] = {}
        # We have to call tools/list once per session again for metadata. This is cheap and keeps the code simple.
        for session in self.sessions:
            allowed_specs = [s for s in self.local_specs if _safe_tool_name(str(s.get("server_label") or "local_mcp")) == session.server_label]
            allowed = set()
            if allowed_specs:
                allowed = set(str(x) for x in (allowed_specs[0].get("allowed_tools") or []) if str(x).strip())
            for tool in session.list_tools():
                original_name = str(tool.get("name") or "").strip()
                if not original_name:
                    continue
                exposed_name = _safe_tool_name(f"{session.server_label}__{original_name}")
                if exposed_name not in self.tool_map:
                    continue
                if allowed and original_name not in allowed and exposed_name not in allowed:
                    continue
                meta[exposed_name] = tool
        self._tool_meta = meta

    def call(self, exposed_name: str, arguments: Any) -> str:
        if exposed_name not in self.tool_map:
            raise ValueError(f"Unknown local MCP function tool: {exposed_name}")
        session, original_name = self.tool_map[exposed_name]
        if isinstance(arguments, str):
            text = arguments.strip()
            args = json.loads(text) if text else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            args = {}
        self.trace.append({"event": "local_mcp_call", "server_label": session.server_label, "tool": original_name, "exposed_name": exposed_name, "arguments": args})
        result = session.call_tool(original_name, args)
        output = _mcp_tool_result_to_text(result)
        self.trace.append({"event": "local_mcp_result", "server_label": session.server_label, "tool": original_name, "exposed_name": exposed_name, "output_preview": output[:1000]})
        return output


def _mcp_tool_result_to_text(result: Any) -> str:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif "data" in item:
                        parts.append(json.dumps(item.get("data"), ensure_ascii=False))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts)
        if "structuredContent" in result:
            return json.dumps(result.get("structuredContent"), ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


def _add_tools_to_body(body: Dict[str, Any], tools: List[Dict[str, Any]]) -> None:
    if not tools:
        return
    existing = body.get("tools")
    if existing is None:
        body["tools"] = list(tools)
    elif isinstance(existing, list):
        body["tools"] = existing + list(tools)
    else:
        raise ValueError("tools field must be a JSON array")


def _extract_chat_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    message = (choices[0] or {}).get("message") or {}
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    return [c for c in calls if isinstance(c, dict)]


def _extract_responses_function_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("function_call", "tool_call") and item.get("name"):
                calls.append(item)
    return calls


def _run_chat_with_local_tools(
    endpoint: str,
    headers: Dict[str, str],
    timeout_sec: int,
    body: Dict[str, Any],
    runtime: _LocalMCPToolRuntime,
    max_tool_rounds: int = 4,
) -> Dict[str, Any]:
    runtime.cache_tool_meta()
    _add_tools_to_body(body, runtime.build_function_tools("chat"))
    response = _post_json(endpoint, body, headers, timeout_sec)
    for round_index in range(max_tool_rounds):
        tool_calls = _extract_chat_tool_calls(response)
        if not tool_calls:
            return response
        choices = response.get("choices") or []
        assistant_msg = (choices[0] or {}).get("message") or {"role": "assistant", "tool_calls": tool_calls}
        body.setdefault("messages", []).append(assistant_msg)
        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name")
            args = fn.get("arguments") or call.get("arguments") or "{}"
            call_id = call.get("id") or f"call_{round_index}_{len(body.get('messages', []))}"
            output = runtime.call(str(name), args)
            body["messages"].append({"role": "tool", "tool_call_id": call_id, "content": output})
        response = _post_json(endpoint, body, headers, timeout_sec)
    response.setdefault("_workflow_knives_warnings", []).append(f"Stopped after max_tool_rounds={max_tool_rounds}")
    return response


def _run_responses_with_local_tools(
    endpoint: str,
    headers: Dict[str, str],
    timeout_sec: int,
    body: Dict[str, Any],
    runtime: _LocalMCPToolRuntime,
    max_tool_rounds: int = 4,
) -> Dict[str, Any]:
    runtime.cache_tool_meta()
    function_tools = runtime.build_function_tools("responses")
    _add_tools_to_body(body, function_tools)
    response = _post_json(endpoint, body, headers, timeout_sec)
    for round_index in range(max_tool_rounds):
        calls = _extract_responses_function_calls(response)
        if not calls:
            return response
        outputs: List[Dict[str, Any]] = []
        for call in calls:
            name = str(call.get("name"))
            args = call.get("arguments") or "{}"
            call_id = call.get("call_id") or call.get("id") or f"call_{round_index}_{len(outputs)}"
            output = runtime.call(name, args)
            outputs.append({"type": "function_call_output", "call_id": call_id, "output": output})
        next_body: Dict[str, Any] = {
            "model": body.get("model"),
            "input": outputs,
            "stream": False,
        }
        # Keep tools visible for providers that do not persist tool definitions across turns.
        if body.get("tools"):
            next_body["tools"] = body.get("tools")
        if body.get("max_output_tokens"):
            next_body["max_output_tokens"] = body.get("max_output_tokens")
        if isinstance(response.get("id"), str):
            next_body["previous_response_id"] = response["id"]
        response = _post_json(endpoint, next_body, headers, timeout_sec)
    response.setdefault("_workflow_knives_warnings", []).append(f"Stopped after max_tool_rounds={max_tool_rounds}")
    return response


def _parse_stop(stop: str) -> Optional[List[str]]:
    text = (stop or "").strip()
    if not text:
        return None
    # Prefer JSON array when provided, otherwise use non-empty lines.
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError("stop must be a JSON string array or newline-separated strings")
        return parsed
    return [line for line in text.splitlines() if line]


def _image_tensor_to_data_urls(image: Any, max_images: int = 1) -> List[str]:
    """Convert ComfyUI IMAGE tensor [B,H,W,C], float 0..1, into PNG data URLs."""
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "Pillow and NumPy are required for image input. They are normally available in ComfyUI."
        ) from e

    if image is None:
        raise ValueError("image is None")

    # ComfyUI IMAGE is normally torch.Tensor [B,H,W,C]. Accept numpy/list-like too.
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,C] or [H,W,C], got {arr.shape}")

    count = max(1, min(int(max_images or 1), int(arr.shape[0])))
    data_urls: List[str] = []

    for i in range(count):
        frame = np.clip(arr[i], 0.0, 1.0)
        frame = (frame * 255.0).round().astype("uint8")

        if frame.shape[-1] == 1:
            pil = Image.fromarray(frame[..., 0], mode="L")
        elif frame.shape[-1] == 3:
            pil = Image.fromarray(frame, mode="RGB")
        elif frame.shape[-1] == 4:
            pil = Image.fromarray(frame, mode="RGBA")
        else:
            raise ValueError(f"Expected 1, 3, or 4 image channels, got {frame.shape[-1]}")

        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_urls.append(f"data:image/png;base64,{b64}")

    return data_urls


def _build_user_content(user_prompt: str, image: Any, image_detail: str, image_max_count: int) -> Any:
    if image is None:
        return user_prompt

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    detail = (image_detail or "auto").strip()

    for data_url in _image_tensor_to_data_urls(image, image_max_count):
        image_url_obj: Dict[str, Any] = {"url": data_url}
        # OpenAI Chat Completions vision currently documents: low, high, original, auto.
        # "omit" is kept for maximum compatibility with stricter local OpenAI-compatible servers.
        if detail != "omit":
            image_url_obj["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url_obj})

    return content

def _build_responses_input(user_prompt: str, image: Any, image_detail: str, image_max_count: int) -> Any:
    """Build OpenAI Responses-style input. Uses input_text/input_image content parts."""
    if image is None:
        return user_prompt

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    detail = (image_detail or "auto").strip()

    for data_url in _image_tensor_to_data_urls(image, image_max_count):
        item: Dict[str, Any] = {"type": "input_image", "image_url": data_url}
        if detail != "omit":
            item["detail"] = detail
        content.append(item)

    return [{"role": "user", "content": content}]


def _apply_token_limit(body: Dict[str, Any], max_tokens: int, max_tokens_field: str, api_mode: str) -> None:
    value = int(max_tokens)
    if api_mode == "responses":
        # Responses API uses max_output_tokens. Keep the UI field name generic; extra_body_json can override.
        body["max_output_tokens"] = value
        return

    if max_tokens_field == "max_completion_tokens":
        body["max_completion_tokens"] = value
    elif max_tokens_field == "both":
        body["max_tokens"] = value
        body["max_completion_tokens"] = value
    else:
        body["max_tokens"] = value


def _build_chat_body(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_field: str,
    thinking: str,
    thinking_api_style: str,
    image: Any,
    image_detail: str,
    image_max_count: int,
    seed: int,
    presence_penalty: float,
    frequency_penalty: float,
    stop: str,
    json_mode: str,
    extra_body_json: str,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": _build_user_content(user_prompt, image, image_detail, image_max_count),
        }
    )

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }

    _apply_token_limit(body, max_tokens, max_tokens_field, "chat")

    if presence_penalty != 0.0:
        body["presence_penalty"] = float(presence_penalty)
    if frequency_penalty != 0.0:
        body["frequency_penalty"] = float(frequency_penalty)
    if seed is not None and int(seed) >= 0:
        body["seed"] = int(seed)

    stop_values = _parse_stop(stop)
    if stop_values:
        body["stop"] = stop_values

    if json_mode == "json_object":
        body["response_format"] = {"type": "json_object"}

    _apply_thinking_params(body, thinking, thinking_api_style)
    body.update(_parse_json_object(extra_body_json, "extra_body_json"))
    return body


def _build_responses_body(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_field: str,
    thinking: str,
    thinking_api_style: str,
    image: Any,
    image_detail: str,
    image_max_count: int,
    seed: int,
    presence_penalty: float,
    frequency_penalty: float,
    json_mode: str,
    extra_body_json: str,
    mcp_tools: Any = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "input": _build_responses_input(user_prompt, image, image_detail, image_max_count),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }

    if (system_prompt or "").strip():
        body["instructions"] = system_prompt

    _apply_token_limit(body, max_tokens, max_tokens_field, "responses")

    if presence_penalty != 0.0:
        body["presence_penalty"] = float(presence_penalty)
    if frequency_penalty != 0.0:
        body["frequency_penalty"] = float(frequency_penalty)
    if seed is not None and int(seed) >= 0:
        body["seed"] = int(seed)

    if json_mode == "json_object":
        # OpenAI Responses-style JSON mode. Provider-specific alternatives can be set via extra_body_json.
        body["text"] = {"format": {"type": "json_object"}}

    _apply_thinking_params(body, thinking, thinking_api_style)
    body.update(_parse_json_object(extra_body_json, "extra_body_json"))
    _merge_mcp_tools_into_body(body, mcp_tools)
    return body



def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return str(message)

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _extract_reasoning_text(choice: Dict[str, Any]) -> str:
    message = choice.get("message") or {}
    candidates = [
        message.get("reasoning_content"),
        message.get("reasoning"),
        message.get("thinking"),
        choice.get("reasoning"),
        choice.get("thinking"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, indent=2)
    return ""

def _extract_responses_text(response: Dict[str, Any]) -> str:
    """Extract the main text from a Responses API-like payload."""
    value = response.get("output_text")
    if isinstance(value, str):
        return value

    parts: List[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str) and item.get("type") in ("output_text", "text"):
                parts.append(item["text"])
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if isinstance(c.get("text"), str) and c.get("type") in ("output_text", "text", None):
                        parts.append(c["text"])
                    elif isinstance(c.get("content"), str):
                        parts.append(c["content"])

    if parts:
        return "\n".join(parts)

    return _extract_message_text(response.get("message") or response.get("response") or response)


def _extract_responses_reasoning_text(response: Dict[str, Any]) -> str:
    """Extract reasoning/thinking snippets from a Responses API-like payload when present."""
    candidates = [response.get("reasoning"), response.get("thinking"), response.get("reasoning_content")]
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("reasoning", "thinking"):
                candidates.append(item)
            for key in ("reasoning", "thinking", "summary"):
                if key in item:
                    candidates.append(item.get(key))
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("reasoning", "thinking", "summary_text"):
                        candidates.append(c)

    parts: List[str] = []
    for value in candidates:
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, dict):
            text = value.get("text") or value.get("summary") or value.get("content")
            if isinstance(text, str) and text.strip():
                parts.append(text)
            elif value:
                parts.append(json.dumps(value, ensure_ascii=False, indent=2))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("summary") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)

    return "\n".join(parts)


def _extract_chat_text_and_reasoning(response: Dict[str, Any]) -> Tuple[str, str]:
    choices = response.get("choices") or []
    if not choices:
        text = _extract_message_text(response.get("message") or response.get("response") or response)
        return text, ""
    choice0 = choices[0]
    return _extract_message_text(choice0.get("message")), _extract_reasoning_text(choice0)


def _apply_thinking_params(body: Dict[str, Any], thinking: str, thinking_api_style: str) -> None:
    """
    Apply optional reasoning/thinking controls.

    Important: there is no fully universal OpenAI-compatible thinking switch.
    Keep thinking_api_style='none' for maximum compatibility.
    """
    if thinking_api_style == "none" or thinking == "default":
        return

    # Map simple UX choices to effort levels.
    effort_map = {
        "off": "none",
        "on": "medium",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
    }
    effort = effort_map.get(thinking, "medium")

    if thinking_api_style == "reasoning_effort":
        body["reasoning_effort"] = effort
    elif thinking_api_style == "reasoning_object":
        body["reasoning"] = {"effort": effort}
    elif thinking_api_style == "ollama_think":
        if thinking == "off":
            body["think"] = False
        elif thinking == "on":
            body["think"] = True
        elif effort in ("low", "medium", "high"):
            body["think"] = effort
        else:
            body["think"] = True
    elif thinking_api_style == "reasoning_effort_and_object":
        body["reasoning_effort"] = effort
        body["reasoning"] = {"effort": effort}


def _origin_from_url(url: str) -> str:
    """Return scheme://host:port from a base or endpoint URL."""
    text = (url or "").strip().rstrip("/")
    if not text:
        raise ValueError("URL is empty")
    parsed = urllib.parse.urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _detect_unload_provider(provider: str, api_base_url: str) -> str:
    """
    Best-effort provider detection.

    There is no standard OpenAI-compatible provider identity endpoint. We only use
    conservative URL/port heuristics and let the user override the result.
    """
    p = (provider or "auto").strip().lower()
    if p in ("none", "off", "disabled"):
        return "none"
    if p in ("lmstudio", "lm_studio", "lm-studio"):
        return "lmstudio"
    if p == "ollama":
        return "ollama"
    if p != "auto":
        return "none"

    parsed = urllib.parse.urlparse((api_base_url or "").strip())
    host = (parsed.hostname or "").lower()
    port = parsed.port
    path = (parsed.path or "").lower()

    # Common defaults: LM Studio OpenAI-compatible server is 1234; Ollama is 11434.
    if port == 1234:
        return "lmstudio"
    if port == 11434:
        return "ollama"

    # Mild hints for non-standard reverse-proxy paths.
    if "lmstudio" in host or "lm-studio" in host or "lmstudio" in path or "lm-studio" in path:
        return "lmstudio"
    if "ollama" in host or "ollama" in path:
        return "ollama"

    return "none"


def _build_common_headers(api_key_env: str, extra_headers_json: str = "") -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ComfyUI-OpenAI-Compatible-LLM-Node/1.0",
    }
    env_name = (api_key_env or "").strip()
    api_key = os.environ.get(env_name, "") if env_name else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers_json:
        headers.update({str(k): str(v) for k, v in _parse_json_object(extra_headers_json, "extra_headers_json").items()})
    return headers


def _get_json(url: str, headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise HttpJsonError(e.code, err_body, url) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to LLM server: {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM server returned non-JSON response: {raw[:1000]}") from e


def _find_lmstudio_loaded_instance_id(models_payload: Dict[str, Any], model: str) -> Optional[str]:
    """Find a loaded LM Studio instance id that corresponds to the requested model."""
    requested = (model or "").strip()
    models = models_payload.get("models", [])
    if not isinstance(models, list):
        return None

    all_loaded_ids: List[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        loaded = item.get("loaded_instances", [])
        if not isinstance(loaded, list) or not loaded:
            continue
        ids = [inst.get("id") for inst in loaded if isinstance(inst, dict) and isinstance(inst.get("id"), str)]
        all_loaded_ids.extend(ids)

        keys_to_match = [
            item.get("key"),
            item.get("display_name"),
            item.get("selected_variant"),
        ]
        variants = item.get("variants")
        if isinstance(variants, list):
            keys_to_match.extend(variants)

        # Exact loaded instance id wins.
        for instance_id in ids:
            if requested and instance_id == requested:
                return instance_id

        # Match by model key/variant/display name, then unload its first loaded instance.
        if requested and any(isinstance(k, str) and k == requested for k in keys_to_match):
            return ids[0] if ids else None

    # Helpful but conservative fallback: if only one model instance is loaded, it is probably the one
    # this just-called node used, even if the OpenAI-compatible model alias differs from LM Studio's key.
    if len(all_loaded_ids) == 1:
        return all_loaded_ids[0]

    return None


def _unload_lmstudio_model(
    api_base_url: str,
    model: str,
    headers: Dict[str, str],
    timeout_sec: int,
) -> Dict[str, Any]:
    """Unload a model in LM Studio via its Native REST API."""
    origin = _origin_from_url(api_base_url)
    endpoint = origin.rstrip("/") + "/api/v1/models/unload"

    # LM Studio's unload endpoint wants a loaded instance id. That id is not practical
    # for a ComfyUI workflow user to know, so always resolve it from /api/v1/models.
    models_endpoint = origin.rstrip("/") + "/api/v1/models"
    models_payload = _get_json(models_endpoint, headers, timeout_sec)
    instance_id = _find_lmstudio_loaded_instance_id(models_payload, model) or ""

    if not instance_id:
        raise ValueError(
            "Could not resolve LM Studio loaded instance id for this model. "
            "If multiple models are loaded, unload manually or make the model name match the LM Studio model key."
        )

    return _post_json(endpoint, {"instance_id": instance_id}, headers, timeout_sec)


def _unload_ollama_model(api_base_url: str, model: str, headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    """Unload a model in Ollama via keep_alive=0."""
    origin = _origin_from_url(api_base_url)
    endpoint = origin.rstrip("/") + "/api/generate"
    model_name = (model or "").strip()
    if not model_name:
        raise ValueError("Ollama unload requires model")
    # Ollama documents unloading by sending an empty prompt and keep_alive=0.
    # stream=false avoids NDJSON streaming and keeps parsing simple.
    return _post_json(endpoint, {"model": model_name, "prompt": "", "keep_alive": 0, "stream": False}, headers, timeout_sec)


def _unload_model_after_call(
    provider: str,
    api_base_url: str,
    model: str,
    headers: Dict[str, str],
    timeout_sec: int,
) -> Tuple[str, Dict[str, Any]]:
    detected = _detect_unload_provider(provider, api_base_url)
    if detected == "lmstudio":
        return detected, _unload_lmstudio_model(api_base_url, model, headers, timeout_sec)
    if detected == "ollama":
        return detected, _unload_ollama_model(api_base_url, model, headers, timeout_sec)
    return detected, {"skipped": True, "reason": "No supported unload provider detected. Set unload_provider explicitly."}


def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise HttpJsonError(e.code, err_body, url) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to LLM server: {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM server returned non-JSON response: {raw[:1000]}") from e


# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------

class OpenAICompatibleLLM:
    """Call an OpenAI-compatible /v1/chat/completions or /v1/responses server and return text."""

    CATEGORY = "LLM/OpenAI Compatible"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning_text", "raw_json", "usage_json", "unload_json")
    OUTPUT_TOOLTIPS = (
        "The assistant's main text response extracted from the first choice.",
        "Reasoning/thinking text if the provider returns it separately; otherwise empty.",
        "The full JSON response from the selected endpoint: Chat Completions or Responses.",
        "The usage object from the response, such as token counts, when provided.",
        "Provider-specific unload result or warning. Empty JSON when unload_after_call is off.",
    )
    DESCRIPTION = "Minimal OpenAI-compatible Chat Completions/Responses caller with optional IMAGE input and optional LM Studio/Ollama unload."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base_url": (
                    "STRING",
                    {
                        "default": "https://api.openai.com/v1",
                        "multiline": False,
                        "tooltip": "Base URL or full endpoint. Use /v1 for the default Chat Completions endpoint. Use /v1/chat/completions to force Chat Completions. Use /v1/responses to try the newer Responses API; if that endpoint is not supported, this node falls back to the sibling /chat/completions endpoint.",
                    },
                ),
                "api_key_env": (
                    "STRING",
                    {
                        "default": "OPENAI_API_KEY",
                        "multiline": False,
                        "tooltip": "Environment variable that contains the Bearer API key/token. Leave empty for local servers that do not require authentication.",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gpt-4o-mini",
                        "multiline": False,
                        "tooltip": "Model ID to send to the API server. Use the exact name expected by OpenAI, LM Studio, Ollama, or your compatible server.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                        "tooltip": "High-level behavior instruction sent as the system message. Leave empty to omit the system message.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "Describe the provided image in detail, focusing on visible subjects, composition, colors, lighting, style, and mood.",
                        "multiline": True,
                        "tooltip": "Main user request sent as the user message. When an IMAGE is connected, this text is sent together with the image.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Controls randomness. Lower values are more deterministic; higher values are more varied. Usually adjust this or top_p, not both.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling cutoff. 1.0 disables the cutoff. Usually leave at 1.0 when tuning temperature.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Maximum number of tokens to generate. Some reasoning-capable servers may count hidden reasoning tokens against this budget.",
                    },
                ),
                "max_tokens_field": (
                    ["max_tokens", "max_completion_tokens", "both"],
                    {
                        "default": "max_tokens",
                        "tooltip": "Which token-limit field to send for Chat Completions. Responses API URLs use max_output_tokens automatically; extra_body_json can override provider-specific details.",
                    },
                ),
                "thinking": (
                    ["default", "off", "on", "minimal", "low", "medium", "high", "xhigh"],
                    {
                        "default": "default",
                        "tooltip": "Reasoning/thinking preference. It is ignored unless thinking_api_style is set to a provider-specific style.",
                    },
                ),
                "thinking_api_style": (
                    [
                        "none",
                        "reasoning_effort",
                        "reasoning_object",
                        "reasoning_effort_and_object",
                        "ollama_think",
                    ],
                    {
                        "default": "none",
                        "tooltip": "Provider-specific way to send thinking controls. Use none for maximum compatibility. The wrong style may be rejected by some servers.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional ComfyUI IMAGE input. Images are encoded as PNG data URLs and attached to the user message.",
                    },
                ),
                "mcp_tools": (
                    "MCP_TOOLS",
                    {
                        "tooltip": "Optional Remote MCP tools list. Only used with Responses API URLs such as /v1/responses. Connect MCP Tools Stack or a preset MCP tool node.",
                    },
                ),
                "image_detail": (
                    ["auto", "low", "high", "original", "omit"],
                    {
                        "default": "auto",
                        "tooltip": "Vision detail hint for image understanding. auto/low/high/original follow OpenAI-style image input options; omit sends no detail field for stricter local servers.",
                    },
                ),
                "image_max_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Maximum number of images to send from a ComfyUI image batch. Start with 1 to avoid large requests.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Optional sampling seed. -1 omits the seed. Determinism is best-effort and depends on the provider/model.",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Penalizes tokens that have already appeared, encouraging new topics. 0.0 is neutral and recommended by default.",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Penalizes repeated tokens based on how often they appear. Raise slightly if the model repeats phrases. 0.0 is neutral.",
                    },
                ),
                "stop": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional. Newline-separated stops or JSON array.",
                        "tooltip": "Optional stop sequences. Generation stops before any listed string is returned. Use newline-separated strings or a JSON array of strings.",
                    },
                ),
                "json_mode": (
                    ["off", "json_object"],
                    {
                        "default": "off",
                        "tooltip": "Ask the API for JSON-object output. Chat Completions sends response_format; Responses sends text.format. The prompt should still explicitly ask for JSON.",
                    },
                ),
                "extra_body_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional JSON object merged into request body, e.g. {\"repetition_penalty\":1.05}",
                        "tooltip": "Advanced escape hatch. A JSON object merged into the request body after normal fields, so it can add or override provider-specific parameters.",
                    },
                ),
                "extra_headers_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional JSON object merged into HTTP headers.",
                        "tooltip": "Advanced escape hatch. A JSON object merged into HTTP headers after the default Content-Type, Accept, User-Agent, and Authorization headers.",
                    },
                ),
                "unload_after_call": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": "If on, attempts to unload the model after the LLM response. Useful for freeing VRAM before downstream image generation.",
                    },
                ),
                "unload_provider": (
                    ["auto", "lmstudio", "ollama", "none"],
                    {
                        "default": "auto",
                        "tooltip": "Provider used for model unload. auto detects common LM Studio/Ollama ports; choose explicitly when using a custom host, port, or reverse proxy.",
                    },
                ),
                "timeout_sec": (
                    "INT",
                    {
                        "default": 120,
                        "min": 1,
                        "max": 3600,
                        "step": 1,
                        "tooltip": "HTTP timeout in seconds for the main request and optional unload calls. Increase if model loading or long responses time out.",
                    },
                ),
            },
        }

    def run(
        self,
        api_base_url: str,
        api_key_env: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_tokens_field: str,
        thinking: str,
        thinking_api_style: str,
        image: Any = None,
        mcp_tools: Any = None,
        image_detail: str = "auto",
        image_max_count: int = 1,
        seed: int = -1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: str = "",
        json_mode: str = "off",
        extra_body_json: str = "",
        extra_headers_json: str = "",
        unload_after_call: str = "off",
        unload_provider: str = "auto",
        timeout_sec: int = 120,
    ) -> Tuple[str, str, str, str, str]:
        api_mode, endpoint, chat_fallback_endpoint = _resolve_api_endpoint(api_base_url)
        headers = _build_common_headers(api_key_env, extra_headers_json)
        remote_tools, local_tools = _split_llm_tools(mcp_tools)
        max_tool_rounds = 4
        tool_trace: List[Dict[str, Any]] = []

        if api_mode != "responses" and remote_tools:
            raise ValueError("Remote MCP tools require a Responses API URL, e.g. http://127.0.0.1:1234/v1/responses")

        def _run_without_local_tools() -> Tuple[Dict[str, Any], str, str]:
            if api_mode == "responses":
                body = _build_responses_body(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_tokens_field=max_tokens_field,
                    thinking=thinking,
                    thinking_api_style=thinking_api_style,
                    image=image,
                    image_detail=image_detail,
                    image_max_count=image_max_count,
                    seed=seed,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    json_mode=json_mode,
                    extra_body_json=extra_body_json,
                    mcp_tools=remote_tools,
                )
                try:
                    resp = _post_json(endpoint, body, headers, int(timeout_sec))
                    return resp, _extract_responses_text(resp), _extract_responses_reasoning_text(resp)
                except Exception as e:
                    if not (chat_fallback_endpoint and _is_responses_endpoint_not_supported_error(e)):
                        raise
                    if remote_tools:
                        raise RuntimeError(
                            "Remote MCP tools require a working Responses API endpoint. "
                            "Chat Completions fallback would silently disable MCP tools."
                        ) from e
                    chat_body = _build_chat_body(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        max_tokens_field=max_tokens_field,
                        thinking=thinking,
                        thinking_api_style=thinking_api_style,
                        image=image,
                        image_detail=image_detail,
                        image_max_count=image_max_count,
                        seed=seed,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        stop=stop,
                        json_mode=json_mode,
                        extra_body_json=extra_body_json,
                    )
                    resp = _post_json(chat_fallback_endpoint, chat_body, headers, int(timeout_sec))
                    txt, rsn = _extract_chat_text_and_reasoning(resp)
                    return resp, txt, rsn
            else:
                chat_body = _build_chat_body(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_tokens_field=max_tokens_field,
                    thinking=thinking,
                    thinking_api_style=thinking_api_style,
                    image=image,
                    image_detail=image_detail,
                    image_max_count=image_max_count,
                    seed=seed,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop=stop,
                    json_mode=json_mode,
                    extra_body_json=extra_body_json,
                )
                resp = _post_json(endpoint, chat_body, headers, int(timeout_sec))
                txt, rsn = _extract_chat_text_and_reasoning(resp)
                return resp, txt, rsn

        if local_tools:
            with _LocalMCPToolRuntime(local_tools) as runtime:
                if api_mode == "responses":
                    body = _build_responses_body(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        max_tokens_field=max_tokens_field,
                        thinking=thinking,
                        thinking_api_style=thinking_api_style,
                        image=image,
                        image_detail=image_detail,
                        image_max_count=image_max_count,
                        seed=seed,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        json_mode=json_mode,
                        extra_body_json=extra_body_json,
                        mcp_tools=remote_tools,
                    )
                    try:
                        response = _run_responses_with_local_tools(endpoint, headers, int(timeout_sec), body, runtime, max_tool_rounds)
                        text = _extract_responses_text(response)
                        reasoning_text = _extract_responses_reasoning_text(response)
                    except Exception as e:
                        if not (chat_fallback_endpoint and _is_responses_endpoint_not_supported_error(e)):
                            raise
                        if remote_tools:
                            raise RuntimeError(
                                "Remote MCP tools require a working Responses API endpoint. "
                                "Chat Completions fallback would silently disable Remote MCP tools."
                            ) from e
                        chat_body = _build_chat_body(
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            max_tokens_field=max_tokens_field,
                            thinking=thinking,
                            thinking_api_style=thinking_api_style,
                            image=image,
                            image_detail=image_detail,
                            image_max_count=image_max_count,
                            seed=seed,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            stop=stop,
                            json_mode=json_mode,
                            extra_body_json=extra_body_json,
                        )
                        response = _run_chat_with_local_tools(chat_fallback_endpoint, headers, int(timeout_sec), chat_body, runtime, max_tool_rounds)
                        text, reasoning_text = _extract_chat_text_and_reasoning(response)
                else:
                    chat_body = _build_chat_body(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        max_tokens_field=max_tokens_field,
                        thinking=thinking,
                        thinking_api_style=thinking_api_style,
                        image=image,
                        image_detail=image_detail,
                        image_max_count=image_max_count,
                        seed=seed,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        stop=stop,
                        json_mode=json_mode,
                        extra_body_json=extra_body_json,
                    )
                    response = _run_chat_with_local_tools(endpoint, headers, int(timeout_sec), chat_body, runtime, max_tool_rounds)
                    text, reasoning_text = _extract_chat_text_and_reasoning(response)
                tool_trace = runtime.trace
        else:
            response, text, reasoning_text = _run_without_local_tools()

        if tool_trace:
            response = dict(response)
            response["_workflow_knives_tool_trace"] = tool_trace

        raw_json = json.dumps(response, ensure_ascii=False, indent=2)
        usage_json = json.dumps(response.get("usage", {}), ensure_ascii=False, indent=2)

        unload_json = "{}"
        if (unload_after_call or "off").strip().lower() == "on":
            try:
                provider_name, unload_response = _unload_model_after_call(
                    unload_provider,
                    api_base_url,
                    model,
                    headers,
                    int(timeout_sec),
                )
                unload_json = json.dumps(
                    {"provider": provider_name, "response": unload_response},
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as e:
                unload_json = json.dumps(
                    {"warning": True, "error": str(e), "policy": "warn"},
                    ensure_ascii=False,
                    indent=2,
                )

        return (text, reasoning_text, raw_json, usage_json, unload_json)

class MCPRemoteTool:
    """Define a generic OpenAI Responses-style Remote MCP tool."""

    CATEGORY = "LLM/OpenAI Compatible/MCP"
    FUNCTION = "build"
    RETURN_TYPES = ("MCP_TOOL",)
    RETURN_NAMES = ("mcp_tool",)
    OUTPUT_TOOLTIPS = ("A single Remote MCP tool definition suitable for the Responses API tools array.",)
    DESCRIPTION = "Define a generic Remote MCP tool for OpenAI-compatible Responses API calls."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_label": (
                    "STRING",
                    {
                        "default": "mcp_server",
                        "multiline": False,
                        "tooltip": "Short label for this MCP server. It appears in tool-call traces.",
                    },
                ),
                "server_url": (
                    "STRING",
                    {
                        "default": "https://example.com/mcp",
                        "multiline": False,
                        "tooltip": "Remote MCP server URL. Supports {{ENV_NAME}} placeholders for flexible URL templates. Prefer headers_json or authorization_env for secrets when the server supports headers.",
                    },
                ),
                "allowed_tools": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional. Newline, comma-separated, or JSON array, e.g. search\nextract",
                        "tooltip": "Optional allowlist of tool names exposed from this MCP server. Leave empty to expose all server tools.",
                    },
                ),
            },
            "optional": {
                "server_description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional description to help the model understand when to use this MCP server.",
                    },
                ),
                "headers_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional JSON object for MCP server headers.",
                        "tooltip": "Optional headers JSON. String values support {{ENV_NAME}} placeholders, e.g. {\"Authorization\":\"Bearer {{MY_TOKEN}}\"}.",
                    },
                ),
                "query_params_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional JSON object for URL query parameters, e.g. {\"apiKey\":\"{{MY_API_KEY}}\",\"transport\":\"sse\"}",
                        "tooltip": "Optional URL query parameters appended to server_url. String values support {{ENV_NAME}} placeholders. Use only when the server requires query parameters.",
                    },
                ),
                "authorization_env": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional environment variable containing an OAuth/access token. When set, its value is sent as the MCP tool authorization field.",
                    },
                ),
            },
        }

    def build(
        self,
        server_label: str,
        server_url: str,
        allowed_tools: str,
        server_description: str = "",
        headers_json: str = "",
        query_params_json: str = "",
        authorization_env: str = "",
    ) -> Tuple[Dict[str, Any]]:
        return (
            _build_remote_mcp_tool(
                server_label=server_label,
                server_url=server_url,
                allowed_tools=allowed_tools,
                server_description=server_description,
                headers_json=headers_json,
                authorization_env=authorization_env,
                query_params_json=query_params_json,
            ),
        )


class MCPTavilyRemoteTool:
    """Preset Remote MCP tool for Tavily Search/Extract/Map/Crawl."""

    CATEGORY = "LLM/OpenAI Compatible/MCP"
    FUNCTION = "build"
    RETURN_TYPES = ("MCP_TOOL",)
    RETURN_NAMES = ("mcp_tool",)
    OUTPUT_TOOLTIPS = ("A Tavily Remote MCP tool definition. Connect to MCP Tools Stack, then to OpenAI Compatible LLM.",)
    DESCRIPTION = "Preset for Tavily's official Remote MCP server. Requires a Tavily API key environment variable."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tavily_api_key_env": (
                    "STRING",
                    {
                        "default": "TAVILY_API_KEY",
                        "multiline": False,
                        "tooltip": "Environment variable containing your Tavily API key.",
                    },
                ),
                "auth_mode": (
                    ["query_param", "authorization_header"],
                    {
                        "default": "query_param",
                        "tooltip": "authorization_header sends Authorization: Bearer ... in tool headers. query_param appends ?tavilyApiKey=... and is a fallback for Tavily setups that require URL query auth.",
                    },
                ),
                "allowed_tools": (
                    "STRING",
                    {
                        "default": "tavily_search",
                        "multiline": True,
                        "tooltip": "Tavily tool allowlist. Common names include tavily_search, tavily_extract, tavily_map, and tavily_crawl. Leave empty to expose all.",
                    },
                ),
            },
            "optional": {
                "default_parameters_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional Tavily DEFAULT_PARAMETERS JSON, e.g. {\"max_results\":5,\"search_depth\":\"basic\"}",
                        "tooltip": "Optional Tavily default parameters header. This is sent as the DEFAULT_PARAMETERS MCP header when provided.",
                    },
                ),
            },
        }

    def build(
        self,
        tavily_api_key_env: str,
        auth_mode: str,
        allowed_tools: str,
        default_parameters_json: str = "",
    ) -> Tuple[Dict[str, Any]]:
        env_name = (tavily_api_key_env or "").strip()
        api_key = os.environ.get(env_name, "") if env_name else ""

        if not api_key:
            raise ValueError(f"Tavily API key environment variable is empty: {env_name!r}")

        base_url = "https://mcp.tavily.com/mcp/"
        headers: Dict[str, str] = {}

        params = (default_parameters_json or "").strip()
        if params:
            # Validate but preserve compact JSON as the Tavily header value.
            parsed_params = _parse_json_object(params, "default_parameters_json")
            headers["DEFAULT_PARAMETERS"] = json.dumps(parsed_params, ensure_ascii=False)

        if auth_mode == "authorization_header":
            server_url = base_url
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            query = urllib.parse.urlencode({"tavilyApiKey": api_key})
            server_url = base_url + "?" + query

        return (
            _build_remote_mcp_tool(
                server_label="tavily",
                server_url=server_url,
                allowed_tools=allowed_tools,
                server_description="Tavily web search, extraction, mapping, and crawling MCP server.",
                headers_json=json.dumps(headers, ensure_ascii=False) if headers else "",
            ),
        )


class MCPDeepWikiRemoteTool:
    """Preset Remote MCP tool for DeepWiki's public MCP server."""

    CATEGORY = "LLM/OpenAI Compatible/MCP"
    FUNCTION = "build"
    RETURN_TYPES = ("MCP_TOOL",)
    RETURN_NAMES = ("mcp_tool",)
    OUTPUT_TOOLTIPS = ("A DeepWiki Remote MCP tool definition for asking questions about public GitHub repositories.",)
    DESCRIPTION = "Preset for DeepWiki's public Remote MCP server. Useful for repository/wiki questions."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "allowed_tools": (
                    "STRING",
                    {
                        "default": "ask_question\nread_wiki_structure",
                        "multiline": True,
                        "tooltip": "DeepWiki tool allowlist. Defaults to the two read-only tools commonly used in OpenAI examples.",
                    },
                ),
            },
            "optional": {
                "server_url": (
                    "STRING",
                    {
                        "default": "https://mcp.deepwiki.com/mcp",
                        "multiline": False,
                        "tooltip": "DeepWiki Remote MCP server URL.",
                    },
                ),
            },
        }

    def build(self, allowed_tools: str, server_url: str = "https://mcp.deepwiki.com/mcp") -> Tuple[Dict[str, Any]]:
        return (
            _build_remote_mcp_tool(
                server_label="deepwiki",
                server_url=server_url,
                allowed_tools=allowed_tools,
                server_description="DeepWiki MCP server for asking questions about public repositories and wiki structure.",
            ),
        )



class MCPLocalCommandTool:
    """Define a local stdio MCP server launched by command for this node run."""

    CATEGORY = "LLM/OpenAI Compatible/MCP"
    FUNCTION = "build"
    RETURN_TYPES = ("MCP_TOOL",)
    RETURN_NAMES = ("mcp_tool",)
    OUTPUT_TOOLTIPS = ("A local stdio MCP server definition. The LLM node starts it only during execution and then terminates it.",)
    DESCRIPTION = "Define a local stdio MCP server command such as npx, uvx, or python. The OpenAI Compatible LLM node runs it as function tools."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_label": (
                    "STRING",
                    {
                        "default": "local",
                        "multiline": False,
                        "tooltip": "Short stable label used to prefix exposed tool names, e.g. time__get_current_time.",
                    },
                ),
                "command": (
                    "STRING",
                    {
                        "default": "uvx",
                        "multiline": False,
                        "tooltip": "Command to launch the stdio MCP server, such as npx, uvx, python, or an absolute executable path.",
                    },
                ),
                "args_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of command arguments. Example: [\"mcp-server-time\", \"--local-timezone=Asia/Tokyo\"].",
                    },
                ),
                "allowed_tools": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional allowlist of MCP tool names. Leave empty to expose all tools from this local MCP server. You may use original names or exposed server_label__tool names.",
                    },
                ),
            },
            "optional": {
                "env_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Extra environment variables for the MCP process as a JSON object. Values are strings.",
                    },
                ),
                "startup_timeout_sec": (
                    "INT",
                    {
                        "default": 15,
                        "min": 1,
                        "max": 300,
                        "step": 1,
                        "tooltip": "Timeout for launching and initializing the MCP server.",
                    },
                ),
                "tool_timeout_sec": (
                    "INT",
                    {
                        "default": 60,
                        "min": 1,
                        "max": 1800,
                        "step": 1,
                        "tooltip": "Timeout for each MCP tool call.",
                    },
                ),
            },
        }

    def build(
        self,
        server_label: str,
        command: str,
        args_json: str,
        allowed_tools: str,
        env_json: str = "{}",
        startup_timeout_sec: int = 15,
        tool_timeout_sec: int = 60,
    ) -> Tuple[Dict[str, Any]]:
        label = _safe_tool_name(server_label or "local")
        cmd = (command or "").strip()
        if not cmd:
            raise ValueError("Local MCP command is required")
        args = _parse_json_array(args_json or "[]", "args_json")
        if not all(isinstance(x, (str, int, float, bool)) for x in args):
            raise ValueError("args_json must be a JSON array of primitive values")
        env = _parse_json_object(env_json or "{}", "env_json")
        allowed = _parse_string_list(allowed_tools, "allowed_tools")
        return (
            {
                "kind": "local_mcp",
                "server_label": label,
                "command": cmd,
                "args": [str(x) for x in args],
                "env": {str(k): str(v) for k, v in env.items()},
                "allowed_tools": allowed,
                "startup_timeout_sec": int(startup_timeout_sec),
                "tool_timeout_sec": int(tool_timeout_sec),
            },
        )

class MCPToolsStack:
    """Collect a variable number of MCP_TOOL inputs into a single MCP_TOOLS list."""

    CATEGORY = "LLM/OpenAI Compatible/MCP"
    FUNCTION = "stack"
    RETURN_TYPES = ("MCP_TOOLS", "STRING")
    RETURN_NAMES = ("mcp_tools", "tools_json")
    OUTPUT_TOOLTIPS = (
        "List of Remote MCP and/or Local MCP tools for the OpenAI Compatible LLM node.",
        "Pretty-printed JSON representation of the tools list for debugging.",
    )
    DESCRIPTION = "Collect any number of MCP_TOOL inputs into one MCP_TOOLS list. Supports both Remote MCP and Local MCP command tools. Inputs are added dynamically in the UI."

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamic inputs are created by the JS extension. Python must accept arbitrary
        # optional input names such as tool_1, tool_2, tool_3, ...
        return {
            "required": {},
            "optional": ContainsAnyDict(),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # Dynamic inputs are not declared in Python ahead of time, so allow the
        # frontend-created MCP_TOOL slots through backend validation.
        return True

    @staticmethod
    def _sort_key(name: str):
        if name == "tools":
            # Backward-compatible hidden support for old workflows; not shown in the UI.
            return (-2, 0, name)
        if name == "tool":
            # Backward-compatible hidden support for old workflows; not shown in the UI.
            return (-1, 0, name)
        prefix = "tool_"
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix.isdigit():
                return (0, int(suffix), name)
        return (1, 0, name)

    def stack(self, **kwargs) -> Tuple[List[Dict[str, Any]], str]:
        out: List[Dict[str, Any]] = []
        for name, value in sorted(kwargs.items(), key=lambda item: self._sort_key(item[0])):
            if not (name == "tool" or name == "tools" or name.startswith("tool_")):
                continue
            if value is None:
                continue
            out.extend(_normalize_mcp_tools(value))
        return (out, json.dumps(out, ensure_ascii=False, indent=2))

class MCPToolsFromJSON:
    """Escape hatch: parse a raw JSON tools array into MCP_TOOLS."""

    CATEGORY = "LLM/OpenAI Compatible/MCP"
    FUNCTION = "parse"
    RETURN_TYPES = ("MCP_TOOLS",)
    RETURN_NAMES = ("mcp_tools",)
    OUTPUT_TOOLTIPS = ("Parsed MCP/LLM tools list.",)
    DESCRIPTION = "Parse a raw JSON array or {tools:[...]} object into MCP_TOOLS. Supports Remote MCP and Local MCP command tool objects."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tools_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of Responses API tool objects, or an object containing a tools array.",
                    },
                ),
            }
        }

    def parse(self, tools_json: str) -> Tuple[List[Dict[str, Any]]]:
        return (_normalize_mcp_tools(tools_json),)


NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleLLM": OpenAICompatibleLLM,
    "MCPRemoteTool": MCPRemoteTool,
    "MCPTavilyRemoteTool": MCPTavilyRemoteTool,
    "MCPDeepWikiRemoteTool": MCPDeepWikiRemoteTool,
    "MCPLocalCommandTool": MCPLocalCommandTool,
    "MCPToolsStack": MCPToolsStack,
    "MCPToolsFromJSON": MCPToolsFromJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleLLM": "OpenAI Compatible LLM",
    "MCPRemoteTool": "MCP Remote Tool",
    "MCPTavilyRemoteTool": "MCP Tavily Remote Tool",
    "MCPDeepWikiRemoteTool": "MCP DeepWiki Remote Tool",
    "MCPLocalCommandTool": "MCP Local Command Tool",
    "MCPToolsStack": "MCP Tools Stack",
    "MCPToolsFromJSON": "MCP Tools from JSON",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
