import os
import aiohttp
import asyncio

# --- Google Gemini configuration (GLOBAL AND ONLY HERE) ---
# Ensure your OPENAI_API_KEY (used here for GEMINI_API_KEY) and OPENAI_BASE_URL (for GEMINI_API_URL)
# are set as environment variables or in a .env file.
# Example: export OPENAI_API_KEY='YOUR_GEMINI_API_KEY_HERE'
# Example: export OPENAI_BASE_URL='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:'
#GEMINI_API_KEY = os.getenv("OPENAI_API_KEY") # Using OPENAI_API_KEY env var for Gemini key
#GEMINI_BASE_URL = os.getenv("OPENAI_BASE_URL") # Using OPENAI_BASE_URL env var for Gemini base 
GEMINI_API_KEY = "AIzaSyCP1l10Tf2xJK15eVROCBgU4r5CxUXjwBs"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"

# --- Data File Path ---
DATA_FILE_PATH = 'banking_current_issues.json' #

async def call_gemini_async(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_s: int = 60,
    retries: int = 3,
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    backoff = 1.0
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.post(
                    GEMINI_URL,
                    headers=headers,
                    params={"key": GEMINI_API_KEY},
                    json=payload,
                    timeout=timeout_s,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    try:
                        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except Exception:
                        return "Error: Unexpected Gemini response format."
            except Exception as e:
                if attempt == retries - 1:
                    return f"Gemini error after {retries} attempts: {e}"
                await asyncio.sleep(backoff)
                backoff *= 2