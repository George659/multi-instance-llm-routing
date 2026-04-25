import httpx

url = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "max_tokens": 32,
    "temperature": 0.0,
}

resp = httpx.post(url, json=payload, timeout=120)
print("status:", resp.status_code)
print(resp.text[:1000])
resp.raise_for_status()
