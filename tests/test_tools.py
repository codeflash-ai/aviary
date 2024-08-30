import json
import pickle
from collections.abc import Callable
from enum import IntEnum, auto
from typing import Any

import pytest
from pydantic import BaseModel, Field
from pytest_subtests import SubTests

from aviary.env import DummyEnv
from aviary.tools import (
    INVALID_TOOL_NAME,
    FunctionInfo,
    Tool,
    ToolCall,
    ToolRequestMessage,
    argref_by_name,
)


def simple() -> None:
    """Doing nothing may be better than doing something."""


def intuitive_arg(x: str) -> float:  # type: ignore[empty-body]
    """Cast the input argument x to a float."""


class StubState(BaseModel):
    """Stub model docstring."""

    defaulted_int: int = Field(default=1, description="A description of the int.")
    required_str: str = Field(description="A description of the str.")


class StubEnum(IntEnum):
    """Stub enum docstring."""

    STUB1 = auto()
    STUB2 = auto()


def many_edge_cases(
    x: int,
    y: None,
    union: int | None,
    pydantic_model: StubState,
    basic_dict: dict[str, int],
    complex_dict: dict[str, tuple[str, int]],
    enum: StubEnum,
    defaulted_str: str = "default",
    defaulted_float: float = 1.0,
) -> None:
    """
    Check using docstrings as partial f-string templates like so: {summary_format}.

    Args:
        x: Yes, I end with a colon :
        y: I am null.
            And despite that there is a multiline argument description.
        union: I am a union and the current year is {current_year}.
        pydantic_model: I am a Pydantic model.
        basic_dict: I am a dictionary with primitive values.
        complex_dict: I am a dictionary with complex values.
        enum: I am an enum.
        defaulted_str: I have a string default value.
        defaulted_float: I have a float default value.
    """


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first number
        b: second number

    Returns:
        I am not yet included, perhaps someday I should be.
    """
    return a + b


def example_fxn(x: int, y: str, z: float) -> None:
    r"""A test function.

    There may be non-summary content.

    \f

    I should be ignored.

    Args:
        x: x
        y: y
        z: z
    """
    assert isinstance(x, int)
    assert isinstance(y, str)
    assert isinstance(z, float)


class TestTool:
    @pytest.mark.parametrize(
        ("fn", "kwargs", "expected"),
        [
            pytest.param(
                simple,
                {},
                {
                    "type": "function",
                    "info": {
                        "name": "simple",
                        "description": (
                            "Doing nothing may be better than doing something."
                        ),
                        "parameters": {
                            "properties": {},
                            "required": [],
                            "type": "object",
                        },
                    },
                },
                id="only-summary",
            ),
            pytest.param(
                intuitive_arg,
                {"allow_empty_param_descriptions": True},
                {
                    "type": "function",
                    "info": {
                        "name": "intuitive_arg",
                        "description": "Cast the input argument x to a float.",
                        "parameters": {
                            "properties": {"x": {"title": "X", "type": "string"}},
                            "required": ["x"],
                            "type": "object",
                        },
                    },
                },
                id="only-summary",
            ),
            pytest.param(
                many_edge_cases,
                {"current_year": 2024},  # Intentionally left format_1 unformatted,
                {
                    "type": "function",
                    "info": {
                        "name": "many_edge_cases",
                        "description": (
                            "Check using docstrings as partial f-string templates like"
                            " so: {summary_format}."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {
                                    "description": "Yes, I end with a colon :",
                                    "title": "X",
                                    "type": "integer",
                                },
                                "y": {
                                    "description": (
                                        "I am null. And despite that there is a"
                                        " multiline argument description."
                                    ),
                                    "title": "Y",
                                    "type": "null",
                                },
                                "union": {
                                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "description": (
                                        "I am a union and the current year is 2024."
                                    ),
                                    "title": "Union",
                                },
                                "pydantic_model": {
                                    "allOf": [{"$ref": "#/$defs/StubState"}],
                                    "description": "I am a Pydantic model.",
                                },
                                "basic_dict": {
                                    "additionalProperties": {"type": "integer"},
                                    "description": (
                                        "I am a dictionary with primitive values."
                                    ),
                                    "title": "Basic Dict",
                                    "type": "object",
                                },
                                "complex_dict": {
                                    "additionalProperties": {
                                        "maxItems": 2,
                                        "minItems": 2,
                                        "prefixItems": [
                                            {"type": "string"},
                                            {"type": "integer"},
                                        ],
                                        "type": "array",
                                    },
                                    "description": (
                                        "I am a dictionary with complex values."
                                    ),
                                    "title": "Complex Dict",
                                    "type": "object",
                                },
                                "enum": {
                                    "allOf": [{"$ref": "#/$defs/StubEnum"}],
                                    "description": "I am an enum.",
                                },
                                "defaulted_str": {
                                    "default": "default",
                                    "description": "I have a string default value.",
                                    "title": "Defaulted Str",
                                    "type": "string",
                                },
                                "defaulted_float": {
                                    "default": 1.0,
                                    "description": "I have a float default value.",
                                    "title": "Defaulted Float",
                                    "type": "number",
                                },
                            },
                            "required": [
                                "x",
                                "y",
                                "union",
                                "pydantic_model",
                                "basic_dict",
                                "complex_dict",
                                "enum",
                            ],
                            "$defs": {
                                "StubEnum": {
                                    "description": "Stub enum docstring.",
                                    "enum": [1, 2],
                                    "title": "StubEnum",
                                    "type": "integer",
                                },
                                "StubState": {
                                    "description": "Stub model docstring.",
                                    "properties": {
                                        "defaulted_int": {
                                            "default": 1,
                                            "description": "A description of the int.",
                                            "title": "Defaulted Int",
                                            "type": "integer",
                                        },
                                        "required_str": {
                                            "description": "A description of the str.",
                                            "title": "Required Str",
                                            "type": "string",
                                        },
                                    },
                                    "required": ["required_str"],
                                    "title": "StubState",
                                    "type": "object",
                                },
                            },
                        },
                    },
                },
                id="many-edge-cases",
            ),
            pytest.param(
                add,
                {},
                {
                    "type": "function",
                    "info": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "properties": {
                                "a": {
                                    "description": "first number",
                                    "title": "A",
                                    "type": "integer",
                                },
                                "b": {
                                    "description": "second number",
                                    "title": "B",
                                    "type": "integer",
                                },
                            },
                            "required": ["a", "b"],
                            "type": "object",
                        },
                    },
                },
                id="with-args-and-returns",
            ),
            pytest.param(
                example_fxn,
                {},
                {
                    "type": "function",
                    "info": {
                        "name": "example_fxn",
                        "description": (
                            "A test function.\n\nThere may be non-summary content."
                        ),
                        "parameters": {
                            "properties": {
                                "x": {
                                    "description": "x",
                                    "title": "X",
                                    "type": "integer",
                                },
                                "y": {
                                    "description": "y",
                                    "title": "Y",
                                    "type": "string",
                                },
                                "z": {
                                    "description": "z",
                                    "title": "Z",
                                    "type": "number",
                                },
                            },
                            "required": ["x", "y", "z"],
                            "type": "object",
                        },
                    },
                },
                id="with-linefeed",
            ),
        ],
    )
    def test_from_function(
        self, fn: Callable, kwargs: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        assert (
            Tool.from_function(fn, **kwargs).model_dump(exclude_none=True) == expected
        )

    @pytest.mark.parametrize(
        ("fn", "kwargs", "expected"),
        [
            (
                example_fxn,
                {},
                """NAME: example_fxn

SYNOPSIS:
    example_fxn(integer x, string y, number z)

DESCRIPTION:
    A test function.

    There may be non-summary content.

PARAMETERS:
    x (integer): x
    y (string): y
    z (number): z""",
            ),
            (
                intuitive_arg,
                {"allow_empty_param_descriptions": True},
                """NAME: intuitive_arg

SYNOPSIS:
    intuitive_arg(string x)

DESCRIPTION:
    Cast the input argument x to a float.

PARAMETERS:
    x (string): No description provided.""",
            ),
        ],
    )
    def test_describe_str(
        self, fn: Callable, kwargs: dict[str, Any], expected: str
    ) -> None:
        tool = Tool.from_function(fn, **kwargs)
        assert tool.info.describe_str().strip() == expected

    def test_describe(self, subtests: SubTests) -> None:
        """Test that describe_xyz functions for FunctionInfo are reasonable."""
        tool = Tool.from_function(example_fxn)

        with subtests.test("Test describe_xml is callable"):
            assert tool.info.describe_xml()

        with subtests.test("Test describe_json is callable"):
            assert tool.info.describe_json()

    def test_serialization_manual(self) -> None:
        # make one manually
        tool = Tool(
            tool_fn=add,
            info=FunctionInfo(
                name="add",
                description="Add two numbers.",
                parameters={
                    "properties": {
                        "a": {
                            "description": "first number",
                            "title": "A",
                            "type": "integer",
                        },
                        "b": {
                            "description": "second number",
                            "title": "B",
                            "type": "integer",
                        },
                    },
                    "required": ["a", "b"],
                    "type": "object",
                },
            ),
        )

        ref = json.loads(r"""{
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "title": "A",
                        "description": "first number"
                    },
                    "b": {
                        "type": "integer",
                        "title": "B",
                        "description": "second number"
                    }
                },
                "required": [
                    "a",
                    "b"
                ]
            }
        }
    }""")
        # make sure it agrees with the reference
        my_dump = json.loads(tool.model_dump_json(exclude_none=True, by_alias=True))
        assert my_dump == ref

        # make one from a function
        tool_fxn = Tool.from_function(add)
        # make sure it serializes correctly
        assert tool_fxn.model_dump_json(
            exclude_none=True, by_alias=True
        ) == tool.model_dump_json(exclude_none=True, by_alias=True)

    @pytest.mark.asyncio
    async def test_arg_types(self) -> None:
        tool = Tool.from_function(example_fxn)

        assert tool.info.parameters.properties["x"]["type"] == "integer"
        assert tool.info.parameters.properties["y"]["type"] == "string"
        assert tool.info.parameters.properties["z"]["type"] == "number"

        calls = [
            ToolCall.from_name(tool.info.name, x=5, y="hi", z=4.2),
        ]
        for call in calls:
            # Call the function to make sure argument types
            # are passed correctly. Private because
            # it doesn't serialize
            tool._tool_fn(**call.function.arguments)

    @pytest.mark.asyncio
    async def test_tool_serialization(
        self, dummy_env: DummyEnv, subtests: SubTests
    ) -> None:
        def get_todo_list(n: int):
            """Get todo list for today.

            Args:
                n: number of items to return
            """
            return "\n".join(["Go for a walk", "Read a book", "Call a friend"][:n])

        tool = Tool.from_function(get_todo_list)

        with subtests.test("pickling"):
            # Check round-trip pickling doesn't break the original Tool
            orig_tool_fn_id = id(tool._tool_fn)
            pickle.loads(pickle.dumps(tool))  # noqa: S301
            assert id(tool._tool_fn) == orig_tool_fn_id

        with subtests.test("serialization then deserialization"):
            tool_copy = Tool(**tool.model_dump(by_alias=True))
            assert tool.type == tool_copy.type
            assert tool.info == tool_copy.info

        dummy_env.tools = [tool]

        with subtests.test("tool call from dump"):
            # Mimic the way an ToolCall might be invoked by an LLM API:
            # the arguments will be strings.
            action = ToolRequestMessage(**{  # noqa: PIE804
                "tool_calls": [
                    {
                        "id": "good_tool_call",
                        "function": {"name": "get_todo_list", "arguments": '{"n": 2}'},
                    },
                    {
                        "id": "bad_tool_call",
                        "function": {
                            "name": "get_todo_list",
                            "arguments": '({"n": 2})',  # NOTE: invalid JSON
                        },
                    },
                ]
            })

            assert action.tool_calls[0].function.arguments == {"n": 2}
            assert action.tool_calls[1].function.name == INVALID_TOOL_NAME

        with subtests.test("tool call from name"):
            tool_call = ToolCall.from_name("get_todo_list", n=2)
            action = ToolRequestMessage(tool_calls=[tool_call])
            new_messages = await dummy_env.exec_tool_calls(action)
            assert new_messages[0].content == "Go for a walk\nRead a book"

        with subtests.test("tool call from tool"):
            tool_call = ToolCall.from_tool(tool, n=2)
            action = ToolRequestMessage(tool_calls=[tool_call])
            new_messages = await dummy_env.exec_tool_calls(action)
            assert new_messages[0].content == "Go for a walk\nRead a book"

        with subtests.test("tool call from tool with no kwargs"):
            tool_call = ToolCall.from_tool(tool, 3)
            action = ToolRequestMessage(tool_calls=[tool_call])
            new_messages = await dummy_env.exec_tool_calls(action)
            assert (
                new_messages[0].content == "Go for a walk\nRead a book\nCall a friend"
            )

        def get_todo_list_no_args():
            """Get todo list for today."""
            return "Go for a walk"

        tool = Tool.from_function(get_todo_list_no_args)
        dummy_env.tools = [tool]

        with subtests.test("tool call from tool with no args and order mismatch"):
            tool_call = ToolCall.from_tool(tool)
            action = ToolRequestMessage(tool_calls=[tool_call])
            new_messages = await dummy_env.exec_tool_calls(action)
            assert new_messages[0].content == "Go for a walk"

            tool_call = ToolCall.from_tool(tool, 1, 10, 30441)
            action = ToolRequestMessage(tool_calls=[tool_call])
            new_messages = await dummy_env.exec_tool_calls(action)
            assert new_messages[0].content == "Go for a walk"


@pytest.mark.asyncio
async def test_argref_by_name() -> None:
    class MyState:
        def __init__(self):
            self.refs = {"foo": 1}

    # Check we can use argref_by_name to add 1 + 2 using a value in refs
    wrapped_add = argref_by_name()(add)
    s = MyState()
    result = wrapped_add("foo", 2, state=s)
    # Now s.refs has a new entry at the below `name`
    name = result.split()[0]
    assert s.refs[name] == 1 + 2

    # Check we can still use argref_by_name without refs
    result = wrapped_add(6, 2, state=s)
    assert s.refs[result.split()[0]] == 6 + 2

    # Check if we use a key name that doesn't exist, we blow up
    with pytest.raises(KeyError, match="not found in state"):
        wrapped_add("bar", 2, state=MyState())

    # Check if state doesn't have refs, we blow up
    with pytest.raises(AttributeError, match="must have a 'refs' attribute"):
        wrapped_add("foo", 2, state="not a state")

    # now try with async and decorator
    @argref_by_name()
    async def async_add(a: int, b: int) -> int:
        """Some docstring."""
        return a + b

    result = await async_add("foo", 2, state=s)
    assert s.refs[result.split()[0]] == 1 + 2
    result = await async_add(6, 2, state=s)
    assert s.refs[result.split()[0]] == 6 + 2

    # now try with lists
    s.refs["bar"] = 7
    result = await async_add("foo", "bar", state=s)
    assert s.refs[result.split()[0]] == 1 + 7

    # try the convenience of comma splitting on key
    result = await async_add("foo,bar", state=s)
    assert s.refs[result.split()[0]] == 1 + 7

    @argref_by_name()
    async def async_list(a: int, b: int) -> list[int]:
        """Some docstring."""
        return [a, b]

    result = await async_list("foo", 2, state=s)
    name1, name2 = (n.split()[0] for n in result.split("\n"))
    assert s.refs[name1] == 1
    assert s.refs[name2] == 2

    @argref_by_name(return_direct=True)
    async def async_list_direct(a: int, b: int) -> list[int]:
        """Some docstring."""
        return [a, b]

    assert await async_list_direct("foo", 2, state=s) == [1, 2]

    # call in context
    tool = Tool.from_function(argref_by_name()(add))

    tool_call = ToolCall.from_tool(tool, "foo", 2)
    action = ToolRequestMessage(tool_calls=[tool_call])
    my_env = DummyEnv()
    my_env.tools = [tool]
    new_messages = await my_env.exec_tool_calls(action, state=MyState())
    assert new_messages[0].content.endswith("3")

    # assert that we can describe the tool
    assert tool.info.describe_str()
    assert (
        "(set via a string key instead of the full object)" in tool.info.describe_str()
    )

    # now try state passing
    @argref_by_name(fxn_requires_state=True)
    async def want_state(a: int, state: MyState) -> int:  # noqa: ARG001
        """Some docstring.

        Args:
            a: first number
            state: the state object
        """
        return 2 * a

    tool = Tool.from_function(want_state)
    action = ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool, "foo")])
    my_env = DummyEnv()
    my_env.tools = [tool]
    await my_env.exec_tool_calls(action, state=MyState())