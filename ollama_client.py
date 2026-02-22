import requests

BASE_URL = "http://localhost:11434"

def generate(prompt, model="llama3.1:8b"):
    url = f"{BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    print("STATUS:", response.status_code)
    print("RAW RESPONSE:", response.text)  # print full response to terminal
    data = response.json()
    print("JSON KEYS:", data.keys())
    return data.get("response") or data.get("message", {}).get("content", "No response found")