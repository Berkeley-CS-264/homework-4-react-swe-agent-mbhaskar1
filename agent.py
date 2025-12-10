"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history list (role, content, timestamp, unique_id)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any
import time
import traceback

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect

# System prompt designed for strong SWE-bench performance
SYSTEM_PROMPT = """You are an expert software engineer tasked with solving GitHub issues. You have access to a codebase and various tools to help you understand, navigate, and modify the code.

## Your Approach
1. **Understand the Issue**: First, carefully read the problem statement to understand what needs to be fixed or implemented.
2. **Explore the Codebase**: Use find_files and search_code to locate relevant files. Use show_file to view code with line numbers.
3. **Locate the Problem**: Find the specific file(s) and function(s) that need to be modified.
4. **Plan Your Fix**: Think through the solution before implementing. Consider edge cases.
5. **Make Minimal Changes**: Only modify what is necessary. Do not refactor unrelated code.
6. **Verify**: After editing, use show_file to verify your changes are correct.
7. **Finish**: Call finish() with a summary when done.

## Important Guidelines
- NEVER modify test files - tests are used for evaluation and must remain unchanged.
- Do NOT run pytest or the full test suite - it takes too long and wastes steps.
- Start by exploring the codebase structure with list_directory and find_files.
- Use search_code to find relevant code patterns mentioned in the issue.
- Use show_file to view files BEFORE making any edits.
- When using replace_in_file, copy content EXACTLY including all whitespace and indentation.
- If replace_in_file fails, re-read the file and try again with exact content.
- Make small, focused changes. One logical change per replace_in_file call.
- After making changes, always verify with show_file that the edit was applied correctly.
- If stuck, try a different approach rather than repeating failed actions.

## Common Patterns
- For bug fixes: Find where the bug occurs, understand the logic, make minimal fix.
- For new features: Find similar existing code, understand the patterns, implement similarly.
- For import errors: Check module structure and fix import statements.

## Tool Tips
- list_directory: Start here to understand project structure
- find_files: Find files by name pattern (e.g., "*.py", "test_*.py")
- search_code: Search for code patterns, function names, class names
- show_file: View file contents with line numbers (essential before editing)
- replace_in_file: Replace exact text - must match exactly including whitespace
- run_bash_cmd: Run shell commands (use sparingly, prefer other tools)
- finish: Call when done with a summary of changes made

When you have completed your fix, call finish() with a brief summary."""


class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history list with unique ids
    - Builds the LLM context from the message list
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message list storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1
        self.next_id: int = 0

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # For trajectory saving
        self.messages: List[Dict[str, Any]] = []

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task)
        self.system_message_id = self.add_message("system", SYSTEM_PROMPT)
        self.user_message_id = self.add_message("user", "")
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

    # -------------------- MESSAGE LIST --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the list.

        The message must include fields: role, content, timestamp, unique_id.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "unique_id": self.next_id
        }
        self.id_to_message.append(message)
        message_id = self.next_id
        self.next_id += 1
        return message_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """
        Update message content by id.
        """
        for message in self.id_to_message:
            if message["unique_id"] == message_id:
                message["content"] = content
                message["timestamp"] = time.time()
                return
        raise ValueError(f"Message with id {message_id} not found")

    def get_context(self) -> str:
        """
        Build the full LLM context from the message list.
        """
        context_parts = []
        for message in self.id_to_message:
            context_parts.append(self.message_id_to_context(message["unique_id"]))
        return "\n".join(context_parts)

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Build the messages list in OpenAI chat format.
        """
        messages = []
        for message in self.id_to_message:
            if message["role"] == "system":
                # Include tool descriptions in system message
                tool_descriptions = []
                for tool in self.function_map.values():
                    signature = inspect.signature(tool)
                    docstring = inspect.getdoc(tool)
                    tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                    tool_descriptions.append(tool_description)

                tool_descriptions_str = "\n".join(tool_descriptions)
                system_content = (
                    f"{message['content']}\n\n"
                    f"--- AVAILABLE TOOLS ---\n{tool_descriptions_str}\n\n"
                    f"--- RESPONSE FORMAT ---\n{self.parser.response_format}"
                )
                messages.append({"role": "system", "content": system_content})
            else:
                messages.append({"role": message["role"], "content": message["content"]})
        return messages

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        for tool in tools:
            self.function_map[tool.__name__] = tool

    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args;
            result (str): A summary of the changes made to fix the issue

        Returns:
            The result passed as an argument. The result is then returned by the agent's run method.
        """
        return result

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message list (with `message_id_to_context`)
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the list
            - If `finish` is called, return the final result
        """
        # Ensure max_steps doesn't exceed 100
        max_steps = min(max_steps, 100)

        # Set the user task message
        self.set_message_content(self.user_message_id, f"Please solve the following GitHub issue:\n\n{task}")

        # Track consecutive errors to avoid infinite loops
        consecutive_errors = 0
        max_consecutive_errors = 5

        # Main ReAct loop
        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")

            try:
                # Build messages for LLM
                messages = self.get_messages_for_llm()

                # Query the LLM
                response = self.llm.generate(messages)

                # Store the response for trajectory
                self.messages.append({
                    "role": "assistant",
                    "content": response,
                    "step": step + 1
                })

                # Parse the function call
                parsed = self.parser.parse(response)

                # Add assistant message to history
                self.add_message("assistant", response)

                # Check if a function was called
                if not parsed["name"]:
                    # No function call found - add error message and continue
                    error_msg = "Error: No function call detected in your response. You MUST call a function using the required format. Review the RESPONSE FORMAT section and try again."
                    self.add_message("user", error_msg)
                    self.messages.append({
                        "role": "user",
                        "content": error_msg,
                        "step": step + 1
                    })
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        return "Agent failed: too many consecutive errors without valid function calls"
                    continue

                function_name = parsed["name"]
                arguments = parsed["arguments"]

                print(f"Function called: {function_name}")
                print(f"Arguments: {list(arguments.keys())}")

                # Check if the function exists
                if function_name not in self.function_map:
                    error_msg = f"Error: Unknown function '{function_name}'. Available functions: {list(self.function_map.keys())}"
                    self.add_message("user", error_msg)
                    self.messages.append({
                        "role": "user",
                        "content": error_msg,
                        "step": step + 1
                    })
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        return "Agent failed: too many consecutive errors"
                    continue

                # Reset error counter on valid function call
                consecutive_errors = 0

                # Execute the function
                try:
                    func = self.function_map[function_name]

                    # Get function signature to handle default arguments
                    sig = inspect.signature(func)
                    params = sig.parameters

                    # Build kwargs, converting types as needed
                    kwargs = {}
                    for param_name, param in params.items():
                        if param_name == 'self':
                            continue
                        if param_name in arguments:
                            value = arguments[param_name]
                            # Try to convert to the expected type
                            if param.annotation != inspect.Parameter.empty:
                                try:
                                    if param.annotation == int:
                                        value = int(value)
                                    elif param.annotation == float:
                                        value = float(value)
                                    elif param.annotation == bool:
                                        value = value.lower() in ('true', '1', 'yes')
                                except (ValueError, AttributeError):
                                    pass
                            kwargs[param_name] = value
                        elif param.default == inspect.Parameter.empty:
                            # Required parameter not provided
                            raise TypeError(f"Missing required argument: {param_name}")

                    result = func(**kwargs)

                    # Check if finish was called
                    if function_name == "finish":
                        print(f"\n--- Agent finished at step {step + 1} ---")
                        return result

                    # Add tool result to history (truncate if too long)
                    result_str = str(result)
                    if len(result_str) > 15000:
                        result_str = result_str[:15000] + "\n... (output truncated)"

                    result_msg = f"Tool '{function_name}' returned:\n{result_str}"
                    self.add_message("user", result_msg)
                    self.messages.append({
                        "role": "user",
                        "content": result_msg,
                        "step": step + 1
                    })

                except Exception as e:
                    error_msg = f"Error executing '{function_name}': {type(e).__name__}: {str(e)}"
                    print(f"Tool error: {error_msg}")
                    self.add_message("user", error_msg)
                    self.messages.append({
                        "role": "user",
                        "content": error_msg,
                        "step": step + 1
                    })

            except Exception as e:
                error_msg = f"Error in step {step + 1}: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                self.add_message("user", error_msg)
                self.messages.append({
                    "role": "user",
                    "content": error_msg,
                    "step": step + 1
                })

        # Max steps reached without finishing
        print(f"\n--- Max steps ({max_steps}) reached ---")
        return "Max steps reached without completing the task."

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        else:
            return f"{header}{content}\n"


def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)


if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()
