"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def _normalized_custom_instructions(custom_instructions: str | None) -> str | None:
    """Return stripped custom instructions, if any."""
    if custom_instructions is None:
        return None

    normalized = custom_instructions.strip()
    return normalized or None


def _custom_instructions_section(custom_instructions: str | None) -> str:
    """Render optional custom instructions for the deriver prompt."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return ""

    return c(
        f"""
        CUSTOM INSTRUCTIONS:
        {normalized_custom_instructions}
        """
    )


def _json_schema_section() -> str:
    """Return the JSON output format instruction for the deriver prompt.

    Separated to avoid f-string brace-escaping complications with the JSON.
    """
    return (
        "OUTPUT FORMAT: Return ONLY a valid JSON object. No prose, no explanation.\n"
        'JSON schema: {"explicit": [{"content": "<observation about {peer_id}>", "salience": <1-10 integer}]}\n'
        '- "content": the extracted fact as a sentence about {peer_id}\n'
        '- "salience": importance of this fact (1=trivial, 10=critical)\n'
        'Example: {"explicit": [{"content": "About {peer_id}: is 25 years old", "salience": 7}, '
        '{"content": "About {peer_id}: birthday is June 21st", "salience": 5}]}'
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: All messages in the range (interleaving messages and new turns combined).

    Returns:
        Formatted prompt string for observation extraction.
    """
    custom_instructions_section = _custom_instructions_section(custom_instructions)
    json_schema = _json_schema_section()

    # Build examples inline using a helper to avoid f-string brace hell
    examples = "\n".join([
        f'- EXPLICIT: "I just had my 25th birthday last Saturday" \u2192 "About {peer_id}: age is 25", "About {peer_id}: birthday is June 21st"',
        f'- EXPLICIT: "I took my dog for a walk in NYC" \u2192 "About {peer_id}: has a dog", "About {peer_id}: lives in or near NYC"',
        f'- EXPLICIT: "{peer_id} attended college" + general knowledge \u2192 "About {peer_id}: completed high school or equivalent"',
    ])

    return c(f"""
Analyze messages from {peer_id} to extract **explicit atomic facts** about them.

[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
{examples}

{json_schema}

{custom_instructions_section}

Messages to analyze:
<messages>
{messages}
</messages>
""")


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=None,
    )
    return estimate_tokens(prompt)


def estimate_deriver_prompt_tokens(custom_instructions: str | None) -> int:
    """Estimate minimal deriver prompt tokens, including custom instructions if present."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return estimate_minimal_deriver_prompt_tokens()

    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=normalized_custom_instructions,
    )
    return estimate_tokens(prompt)