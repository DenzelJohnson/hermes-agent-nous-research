"""
Hermes tool call parser.

Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
Based on VLLM's Hermes2ProToolParser.extract_tool_calls()
"""

import json
import re
import uuid
from typing import Any, List, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser


@register_parser("hermes")
class HermesToolCallParser(ToolCallParser):
    """
    Parser for Hermes-format tool calls.

    Matches <tool_call>...</tool_call> tags containing JSON with "name" and "arguments".
    Also handles unclosed <tool_call> at end-of-string (truncated generation).
    """

    # Matches both closed and unclosed tool_call tags
    XML_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)", re.DOTALL
    )
    TOOLCALL_PREFIX_PATTERN = re.compile(r"(?is)\btoolcall>\s*")

    @staticmethod
    def _close_open_json_structures(payload: str) -> str:
        """Best-effort repair for truncated TOOLCALL> JSON payloads.

        Some open models emit a complete JSON array/object but drop the final
        closing bracket(s) at the end of the turn. We keep the repair narrow:
        only close still-open container delimiters while respecting quoted
        strings and escapes.
        """
        if not payload:
            return payload

        stack: List[str] = []
        in_string = False
        escape = False

        for ch in payload:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()

        if not stack:
            return payload

        closers = {"{": "}", "[": "]"}
        return payload + "".join(closers[ch] for ch in reversed(stack))

    @staticmethod
    def _tool_call_from_payload(item: Any) -> Optional[ChatCompletionMessageToolCall]:
        if not isinstance(item, dict):
            return None

        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name") or item.get("name")
            arguments = function.get("arguments", item.get("arguments", {}))
        else:
            name = item.get("name")
            arguments = item.get("arguments", {})

        if not isinstance(name, str) or not name.strip():
            return None

        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments or {}, ensure_ascii=False)

        return ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=Function(
                name=name.strip(),
                arguments=arguments_str,
            ),
        )

    def _parse_toolcall_prefix_payload(self, text: str) -> ParseResult:
        match = self.TOOLCALL_PREFIX_PATTERN.search(text)
        if not match:
            return text, None

        payload = text[match.end():].strip()
        if not payload:
            return text, None

        decoder = json.JSONDecoder()
        parsed_obj = None

        for candidate in (payload, self._close_open_json_structures(payload)):
            try:
                parsed_obj, _ = decoder.raw_decode(candidate)
                break
            except json.JSONDecodeError:
                continue

        if parsed_obj is None:
            return text, None

        raw_calls = parsed_obj if isinstance(parsed_obj, list) else [parsed_obj]
        tool_calls: List[ChatCompletionMessageToolCall] = []
        for item in raw_calls:
            tc = self._tool_call_from_payload(item)
            if tc is not None:
                tool_calls.append(tc)

        if not tool_calls:
            return text, None

        content = text[: match.start()].strip()
        return content if content else None, tool_calls

    def parse(self, text: str) -> ParseResult:
        if "<tool_call>" not in text and not self.TOOLCALL_PREFIX_PATTERN.search(text):
            return text, None

        prefixed_content, prefixed_calls = self._parse_toolcall_prefix_payload(text)
        if prefixed_calls:
            return prefixed_content, prefixed_calls

        try:
            matches = self.XML_PATTERN.findall(text)
            if not matches:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []
            for match in matches:
                # match is a tuple: (closed_content, unclosed_content)
                raw_json = match[0] if match[0] else match[1]
                if not raw_json.strip():
                    continue

                tc_data = json.loads(raw_json)
                if "name" not in tc_data:
                    continue
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name=tc_data["name"],
                            arguments=json.dumps(
                                tc_data.get("arguments", {}), ensure_ascii=False
                            ),
                        ),
                    )
                )

            if not tool_calls:
                return text, None

            # Content is everything before the first <tool_call> tag
            content = text[: text.find("<tool_call>")].strip()
            return content if content else None, tool_calls

        except Exception:
            return text, None
