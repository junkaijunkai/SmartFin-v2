# system-design.md

## LLM Invocation and Function Calling
The division of work should be clear: LLM invocation is responsible for intent interpretation and decision making, while function calling executes tasks.

### Tool Definition
Define a Client-Executed Tool using JSON Schema. 

Key fields:
| Parameter | Note |
| --- | --- |
| name | The unique identifier of the tool. Naming rule: ^[a-zA-Z0-9_-]{1,64}$ |
| description | Explain the function of the tool, when it is invoked, and what behaviors it exhibits. |
| input_schema | A JSON object, defining the expected parameters by the tool, for the tool, specifying the structure and types of inputs the LLM will provide in tool_use content blocks. |
| input_examples | Optional. Providing concrete example input objects to help LLM understand how to call the tool, particularly useful for complex tools with nested objects, optional parameters, or format-sensitive inputs. |

#### Best Practices
- Be specific in `description`, with at least 3~4 sentences. It should include: 
  <ol type='a'>
  <li>Tool functionality</li>
  <li>When to use (and when not to use)</li>
  <li>The meaning of each parameter and how they affect the tool's behavior</li>
  <li>Any important considerations or limitations</li>
  </ol>
- Prioritize `description` over `input_examples`. Add `input_examples` for a tool designed for a complex task
- Be meaningful in `name`. It should clearly convey the tool's function.
- Use distinct namespaces: when tools span multiple services or resources, prefix names with the service (e.g., `github_list_prs`, `slack_send_message`) to make tool selection unambiguous.
- Group operations with tight correlations within one tool definition to reduce selection ambiguity. (e.g., Rather than creating separate tools like `create_pr`, `review_pr`, and `merge_pr`, you should group them into a single tool with an `action` parameter)
- Return high-signal responses. For tool responses, return semantic, stable identifiers (e.g., slugs or UUIDs) rather than opaque internal references, and include only the fields the LLM needs to reason about its next step.

Example:
```json
{
  "name": "get_weather",
  "description": "Get the current weather in a given location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'"
      }
    },
    "required": ["location"]
  }
}
```

## Regrex and Fallbacks
User messages are diverse. Situations where language interpretation is required should be delegated to the LLM rather than handled with hand-crafted regex. Regex patterns can't be general solutions, hence they should exist only as a fallback when the LLM call fails.

Fallbacks should also cover cases where the LLM correctly understands the user's intent and all required data is present, but tool calling is skipped, like model non-compliance or response format issues. In these cases, fallbacks should be triggered and execute the intended task directly using the parsed parameters.

Examples:
>**Goal Planning Agent** — tool call fallback: The user says "I want to save $1000 by the end of June for an iPhone". The LLM correctly calls the `extract_goal_tool`, but returns the final_answer instead of invoking the `create_goal_tool`. The fallback should verify the extracted parameters are complete and unambiguous, then create the FinancialGoal object directly rather than surfacing an error or re-prompting the user.<br>
>**Expense Analysis Agent** — transaction input parsing: Users may describe transactions as "$48 Grab ride yesterday", "spent 120 on groceries at NTUC", or "lunch 15 bucks". The LLM should extract amount, category, merchant, and date from free-form text. DO NOT accumulate regrex parsing patterns for each variation.

## Agent Working Paradigm

Prioritize simplified design, any introduction of compelxity should be based on actual engineering requirements.

Example:
>Not every agent needs ReAct and tool calling. Apply them only where the task is open-ended or requires dynamic decision-making at runtime.<br>
For agents such as `Health Assessment` and `Anomaly Detection` who have concrete task execution chains, ReAct loop and Function Calling aren't necessary.
