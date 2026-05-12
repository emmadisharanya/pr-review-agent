import hmac, hashlib, json, os
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from github_client import get_pr_diff, post_review_comment
from groq import Groq

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify_signature(payload: bytes, signature: str) -> bool:
    secret = os.getenv("WEBHOOK_SECRET", "").encode()
    expected = "sha256=" + hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

def review_diff(diff: dict) -> str:
    files_text = ""
    for f in diff["files"]:
        files_text += f"\nFile: {f['filename']} ({f['status']})\n{f['patch']}\n"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a senior software engineer reviewing a pull request. Be concise and specific."
            },
            {
                "role": "user",
                "content": f"Review this PR titled '{diff['title']}'.\n\nChanges:\n{files_text}\n\nPoint out bugs, security issues, and improvements. Be specific about file and line."
            }
        ]
    )
    return response.choices[0].message.content

@app.post("/webhook")
async def webhook(request: Request):
    payload_bytes = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(payload_bytes, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")
    payload = json.loads(payload_bytes)
    event = request.headers.get("X-GitHub-Event")
    if event == "pull_request" and payload.get("action") in ["opened", "synchronize"]:
        repo_name = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]
        diff = get_pr_diff(repo_name, pr_number)
        review = review_diff(diff)
        post_review_comment(repo_name, pr_number, review)
        print("\n" + "="*50, flush=True)
        print(f"PR #{pr_number}: {payload['pull_request']['title']}", flush=True)
        print("="*50, flush=True)
        print(review, flush=True)
        print("="*50 + "\n", flush=True)
        return {"status": "processed", "pr": pr_number}
    return {"status": "ignored"}

@app.get("/health")
async def health():
    return {"status": "ok"}