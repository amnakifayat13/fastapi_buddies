import requests

url = "http://127.0.0.1:8000/voice-agent"
files = {"audio": open(r"C:\Amna\party_with_buddies\fastapi_backend\test5_audio.wav", "rb")}


res = requests.post(url, files=files)

print("Status:", res.status_code)
print("Response JSON:", res.json())
