## Comments

Keep comments rare and information-dense.

- Do not add comments by default.
- Do not explain code that is already obvious from the implementation.
- Never narrate the implementation line-by-line.
- Do not add comments for simple assignments, conditionals, loops, function calls, or standard library usage.
- Prefer better naming and structure over explanatory comments.
- Comments should explain information that cannot be expressed clearly in the code.

Add a comment only when it explains one of:

- non-obvious intent or rationale
- an important business rule
- an external API/protocol assumption
- a performance or security constraint
- an edge case that is easy to miss
- a workaround for a known bug or limitation

Keep implementation comments to one or two concise lines whenever possible.

Do not write multi-paragraph comments or verbose explanatory blocks.

For public APIs, use docstrings/documentation when the API contract, parameters, return behavior, exceptions, or usage constraints need to be documented.

When modifying existing code:

- Preserve useful existing comments.
- Remove comments that are made obsolete by the change.
- Do not rewrite comments merely for stylistic reasons.
- Do not perform unrelated comment cleanup.

Default assumption: if the code is understandable without the comment, do not write the comment.
