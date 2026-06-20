# code-principle.md

## General Guidelines
### Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First
**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

### Test-Driven Development
For new services, follow TDD as the following:
1. **Red** – Write a failing test for the desired behavior before writing implementation code. Confirm it fails for the right reason.
2. **Green** – Write the minimal code needed to pass the test. No extra features. Run the full suite to confirm nothing else breaks.
3. **Refactor** – Clean up code/tests without changing behavior. Re-run tests to confirm they still pass.

For refactoring an existing module: Check the TDD first.

**Do Not** write test cases to prove your code, it doesn't make sense.


### Edge Cases and Fallbacks
Do consider edge cases all the time as they are a very IMPORTANT part of the design or modification of a service, and degradation/fallback plans should be designed to handle edge cases gracefully.