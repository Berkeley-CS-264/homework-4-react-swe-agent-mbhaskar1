class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"
    VALUE_SEP = "----VALUE----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
{VALUE_SEP}
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
{VALUE_SEP}
arg2_value (can be multiline)
...
{END_CALL}

DO NOT CHANGE ANY TEST! AS THEY WILL BE USED FOR EVALUATION.
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        """
        result = {
            "thought": "",
            "name": "",
            "arguments": {}
        }

        # Use rfind to find the LAST occurrence of markers (in case they appear in reasoning)
        end_idx = text.rfind(self.END_CALL)
        if end_idx == -1:
            # No function call found - return empty result with full text as thought
            result["thought"] = text.strip()
            return result

        begin_idx = text.rfind(self.BEGIN_CALL)
        if begin_idx == -1 or begin_idx > end_idx:
            # Malformed - no begin marker or it's after end marker
            result["thought"] = text.strip()
            return result

        # Extract thought (everything before BEGIN_CALL)
        result["thought"] = text[:begin_idx].strip()

        # Extract the function call block (between BEGIN and END)
        call_block = text[begin_idx + len(self.BEGIN_CALL):end_idx].strip()

        if not call_block:
            return result

        # Split by ARG_SEP to get function name and arguments
        parts = call_block.split(self.ARG_SEP)

        # First part is the function name
        result["name"] = parts[0].strip()

        # Remaining parts are argument pairs (arg_name, VALUE_SEP, arg_value)
        for i in range(1, len(parts)):
            arg_part = parts[i]

            # Split by VALUE_SEP to get arg name and value
            value_sep_idx = arg_part.find(self.VALUE_SEP)
            if value_sep_idx == -1:
                # Malformed argument - skip it
                continue

            arg_name = arg_part[:value_sep_idx].strip()
            arg_value = arg_part[value_sep_idx + len(self.VALUE_SEP):].strip()

            if arg_name:
                result["arguments"][arg_name] = arg_value

        return result
