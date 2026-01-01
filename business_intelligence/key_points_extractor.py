from business_intelligence.llm_client import get_llm_client

SYSTEM_PROMPT = """
You are a senior business analyst.

Analyze the following meeting transcript and extract structured business insights.
Be concise, factual, and professional.
"""

USER_PROMPT_TEMPLATE = """
Meeting Transcript:
{conversation}

Tasks:
1. List key discussion points
2. List decisions made (if any)
3. List action items (if any)
4. Identify meeting intent (Informational / Planning / Decision-making / Negotiation / Conflict)
5. Determine overall business sentiment (Positive / Neutral / Negative)

Respond strictly in the following JSON format:

{{
  "key_points": [],
  "decisions": [],
  "action_items": [],
  "meeting_intent": "",
  "sentiment": ""
}}
"""

def extract_business_key_points(conversation_text: str):
    client = get_llm_client()
    if client is None:
        return None  # LLM not enabled

    prompt = USER_PROMPT_TEMPLATE.format(conversation=conversation_text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    try:
        import json
        return json.loads(content)
    except Exception:
        return {
            "raw_output": content
        }
