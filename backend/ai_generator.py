import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **Outline tool** (`get_course_outline`): use for questions about a course's outline, structure, or list of lessons
  - Always return the course title, the course link, and every lesson with its number and title
  - Format: course title and link on their own lines, then a "Lessons (N):" line, then a blank line, then a bulleted list with one lesson per line as "- Lesson <number>: <title>" (including Lesson 0 if present). Never use numbered-list syntax for lessons, since Markdown renumbers it
- **Search tool** (`search_course_content`): use **only** for questions about specific course content or detailed educational materials
- **Up to 2 sequential tool calls per query**: you may call a tool, review the result, then call another if needed (e.g. get a lesson title from the outline, then search that topic)
  - Use a second call only when the first result is insufficient or the question has multiple parts
  - Do not repeat a search that already answered the question
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer (up to two rounds)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    # Max sequential tool rounds per query; the request after the last round omits tools
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "max_tokens": 2000
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        messages = [{"role": "user", "content": query}]

        # Get response from Claude
        response = self._call_api(messages, system_content, tools)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(
                response, messages, system_content, tools, tool_manager
            )

        # Return direct response
        return self._extract_text(response)

    def _call_api(self, messages: List[Dict[str, Any]], system: str, tools: Optional[List] = None):
        """Single place that calls the API; tools are offered only when given"""
        params = {
            **self.base_params,
            # Copy so later appends don't alter what this call was made with
            "messages": list(messages),
            "system": system
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**params)

    @staticmethod
    def _extract_text(response) -> str:
        """Return the first text block; content may start with a thinking block"""
        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text:
            return "Sorry, I couldn't generate a response. Please try again."
        return text
    
    def _handle_tool_execution(self, initial_response, messages: List[Dict[str, Any]],
                               system: str, tools: Optional[List], tool_manager):
        """
        Run up to MAX_TOOL_ROUNDS sequential tool rounds, each followed by a separate API request.

        Stops when Claude's response has no tool_use, the round cap is reached, or a tool fails.
        The request after the last round omits tools so Claude must answer in text.

        Args:
            initial_response: The response containing the first tool use request
            messages: Conversation so far (the user query)
            system: System prompt, reused on every request
            tools: Tool definitions offered while more rounds are allowed
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = list(messages)
        response = initial_response
        rounds = 0

        while response.stop_reason == "tool_use" and any(b.type == "tool_use" for b in response.content):
            messages.append({"role": "assistant", "content": response.content})
            tool_results, failed = self._execute_tool_blocks(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})
            rounds += 1

            last_round = failed or rounds >= self.MAX_TOOL_ROUNDS
            response = self._call_api(messages, system, None if last_round else tools)
            if last_round:
                break

        return self._extract_text(response)

    @staticmethod
    def _execute_tool_blocks(response, tool_manager):
        """Execute every tool_use block; returns (tool_result blocks, whether any failed)"""
        tool_results = []
        failed = False
        for content_block in response.content:
            if content_block.type == "tool_use":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                }
                try:
                    result_block["content"] = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    # Report the failure to Claude instead of failing the whole request
                    result_block["content"] = f"Tool error: {e}"
                    result_block["is_error"] = True
                    failed = True
                tool_results.append(result_block)
        return tool_results, failed