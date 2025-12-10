from utils import get_sb_environment
import subprocess
import swebench

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
        self.instance = instance  # Store instance for test execution

    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run a shell command and return the output. Returns both stdout and stderr.
        Use this for commands like: ls, cat, git, pip, etc.
        Note: Prefer using specialized tools (show_file, search_code, etc.) over bash when possible.

        Args;
            command (str): the shell command to run

        Returns:
            The combined stdout and stderr output of the command
        """
        try:
            output = self.env.execute(command)

            # Handle case where execute returns a dict instead of string
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")

            return output

        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            return f"Command timed out. Partial output:\n{output}"
        except TimeoutError:
            return "Command timed out."
        except ValueError as e:
            # Return the error output instead of raising - this lets the agent see what went wrong
            return f"Command returned non-zero exit code. Output:\n{str(e)}"
        except Exception as e:
            return f"Error running command: {type(e).__name__}: {str(e)}"

    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            patch_output = self.env.execute("git add -A && git diff --cached")

            # Handle case where execute returns a dict instead of string
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", "") or patch_output.get("stdout", "")

            if patch_output and patch_output.strip():
                return patch_output
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"

    # -------------------- CUSTOM TOOLS FOR BETTER PERFORMANCE --------------------

    def show_file(self, file_path: str, start_line: int = 1, end_line: int = -1) -> str:
        """
        Display the contents of a file with line numbers. ALWAYS use this before editing a file.

        Args;
            file_path (str): the path to the file to display
            start_line (int): the starting line number (1-indexed, default: 1)
            end_line (int): the ending line number (inclusive, -1 means show 200 lines from start, default: -1)

        Returns:
            The file contents with line numbers prefixed to each line
        """
        try:
            # Read the file
            output = self.env.execute(f"cat -n '{file_path}'")
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")

            lines = output.split('\n')

            # Filter by line range if specified
            start_line = max(1, int(start_line))
            end_line = int(end_line)

            if end_line == -1:
                end_line = start_line + 199  # Show 200 lines by default

            # Line numbers in cat -n start at 1, we need to filter
            filtered_lines = []
            for line in lines:
                # Parse line number from cat -n output (format: "     1\tcode")
                stripped = line.lstrip()
                if stripped:
                    parts = stripped.split('\t', 1)
                    if parts[0].isdigit():
                        line_num = int(parts[0])
                        if start_line <= line_num <= end_line:
                            filtered_lines.append(line)
                    else:
                        filtered_lines.append(line)

            if not filtered_lines:
                return f"No content found in lines {start_line}-{end_line} of {file_path}"

            return '\n'.join(filtered_lines)
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"

    def replace_in_file(self, file_path: str, old_content: str, new_content: str) -> str:
        """
        Replace a specific string/code block in a file with new content.
        IMPORTANT: The old_content must match EXACTLY, including all whitespace and indentation.
        Use show_file first to see the exact content, then copy it precisely.

        Args;
            file_path (str): the path to the file to modify
            old_content (str): the exact content to find and replace (must match exactly including whitespace)
            new_content (str): the new content to replace it with

        Returns:
            A message indicating success or failure of the replacement
        """
        try:
            # First read the file
            output = self.env.execute(f"cat '{file_path}'")
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")

            # Check if old_content exists in the file
            if old_content not in output:
                # Try to find similar content to help debug
                lines = old_content.split('\n')
                first_line = lines[0].strip() if lines else ""
                if first_line and first_line in output:
                    return f"Error: Could not find exact match in {file_path}. The first line '{first_line}' exists but the full content doesn't match exactly. Check whitespace and indentation. Use show_file to see the exact content."
                return f"Error: Could not find the specified content in {file_path}. The content must match exactly including all whitespace and indentation. Use show_file to view the exact content first."

            # Count occurrences
            count = output.count(old_content)
            if count > 1:
                return f"Error: Found {count} occurrences of the content in {file_path}. Please provide more surrounding context to make the match unique."

            # Perform the replacement
            new_file_content = output.replace(old_content, new_content, 1)

            # Write the new content back using a heredoc to handle special characters
            # Use a unique delimiter
            delimiter = "REPLACE_EOF_MARKER_12345"

            # Escape any occurrences of the delimiter in the content
            safe_content = new_file_content.replace(delimiter, delimiter + "_ESCAPED_")

            write_cmd = f"cat << '{delimiter}' > '{file_path}'\n{safe_content}\n{delimiter}"
            self.env.execute(write_cmd)

            return f"Successfully replaced content in {file_path}"
        except Exception as e:
            return f"Error replacing content in '{file_path}': {e}"

    def search_code(self, pattern: str, path: str = ".", file_pattern: str = "") -> str:
        """
        Search for a pattern in code files using grep. Returns matching lines with file paths and line numbers.
        Use this to find function definitions, class names, error messages, etc.

        Args;
            pattern (str): the regex pattern to search for (e.g., "def my_function", "class MyClass")
            path (str): the directory path to search in (default: current directory ".")
            file_pattern (str): optional file pattern to filter (e.g., "*.py" for Python files only)

        Returns:
            Matching lines in format "file:line_number:content", limited to first 50 matches
        """
        try:
            # Build the grep command
            if file_pattern:
                cmd = f"grep -rn --include='{file_pattern}' '{pattern}' {path} 2>/dev/null | head -50"
            else:
                cmd = f"grep -rn '{pattern}' {path} 2>/dev/null | head -50"

            output = self.env.execute(cmd)
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")

            if not output.strip():
                return f"No matches found for pattern '{pattern}' in {path}"

            return output
        except Exception as e:
            # grep returns exit code 1 when no matches found
            return f"No matches found for pattern '{pattern}' in {path}"

    def find_files(self, name_pattern: str, path: str = ".") -> str:
        """
        Find files by name pattern. Useful for locating specific files in the codebase.
        Use this to find files like "models.py", "test_*.py", "*.js", etc.

        Args;
            name_pattern (str): the filename pattern to search for (e.g., "*.py", "test_*.py", "models.py")
            path (str): the directory path to search in (default: current directory ".")

        Returns:
            List of matching file paths, limited to first 50 matches
        """
        try:
            cmd = f"find {path} -type f -name '{name_pattern}' 2>/dev/null | grep -v __pycache__ | grep -v '.git/' | head -50"
            output = self.env.execute(cmd)
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")

            if not output.strip():
                return f"No files found matching pattern '{name_pattern}' in {path}"

            return output
        except Exception as e:
            return f"Error finding files: {e}"

    def list_directory(self, path: str = ".") -> str:
        """
        List contents of a directory showing files and subdirectories.
        Start with this to understand the project structure.

        Args;
            path (str): the directory path to list (default: current directory ".")

        Returns:
            Directory listing showing files and subdirectories
        """
        try:
            cmd = f"ls -la '{path}'"
            output = self.env.execute(cmd)
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")

            return output
        except Exception as e:
            return f"Error listing directory '{path}': {e}"

    def create_file(self, file_path: str, content: str) -> str:
        """
        Create a new file with the given content. Use this to add new files to the project.

        Args;
            file_path (str): the path where the file should be created
            content (str): the content to write to the file

        Returns:
            A message indicating success or failure
        """
        try:
            # Use a heredoc with a unique delimiter
            delimiter = "CREATE_EOF_MARKER_12345"
            safe_content = content.replace(delimiter, delimiter + "_ESCAPED_")

            # Create parent directories if they don't exist
            self.env.execute(f"mkdir -p $(dirname '{file_path}')")

            write_cmd = f"cat << '{delimiter}' > '{file_path}'\n{safe_content}\n{delimiter}"
            self.env.execute(write_cmd)

            return f"Successfully created file {file_path}"
        except Exception as e:
            return f"Error creating file '{file_path}': {e}"

    def view_around_line(self, file_path: str, line_number: int, context: int = 10) -> str:
        """
        View lines around a specific line number in a file. Useful after finding a match with search_code.

        Args;
            file_path (str): the path to the file
            line_number (int): the center line number to view around
            context (int): number of lines to show before and after (default: 10)

        Returns:
            The file contents around the specified line with line numbers
        """
        start = max(1, int(line_number) - int(context))
        end = int(line_number) + int(context)
        return self.show_file(file_path, start, end)


class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output

    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        return self.execute(command)
