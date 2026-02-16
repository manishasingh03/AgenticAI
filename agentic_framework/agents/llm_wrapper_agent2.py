#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# llm_wrapper.py

import aiohttp
import asyncio
import os
from openai import AsyncOpenAI


GEMINI_API_KEY = "AIzaSyCP1l10Tf2xJK15eVROCBgU4r5CxUXjwBs"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# OPENAI settings
#openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_gemini_api(system_prompt, user_prompt):
    headers = { "Content-Type": "application/json" }
    prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        }
    }

    async with aiohttp.ClientSession() as session:
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            if "candidates" in data:
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Error: No response from Gemini."
            
async def call_openai_api(system_prompt, user_prompt):
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

async def call_llm(system_prompt, user_prompt, model="gemini"):
    if model == "gemini":
        return await call_gemini_api(system_prompt, user_prompt)
    elif model == "openai":
        return await call_openai_api(system_prompt, user_prompt)
    else:
        raise ValueError("Unsupported LLM provider. Choose 'gemini' or 'openai'.")

