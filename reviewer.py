import os
from github import Github
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_pr_diff(repo_name, pr_number):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    files = []
    for f in pr.get_files():
        files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": f.patch or ""
        })
    return {
        "title": pr.title,
        "body": pr.body,
        "files": files
    }

def review_diff(diff):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    files_text = ""
    for f in diff["files"]:
        files_text += f"\nFile: {f['filename']} ({f['status']})\n{f['patch']}\n"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a senior software engineer reviewing a pull request.
Format your review exactly like this, with each issue on its own line:

🔴 CRITICAL: <file>:<line> - <description of bug or security issue>
🟡 SUGGESTION: <file>:<line> - <description of improvement>
🔵 NITPICK: <file>:<line> - <minor style or readability issue>

Only use these three labels. Be specific about file and line. If no issues found in a category, skip it."""
            },
            {
                "role": "user",
                "content": f"Review this PR titled '{diff['title']}'.\n\nChanges:\n{files_text}"
            }
        ]
    )
    return response.choices[0].message.content

def save_review(repo_name, pr_number, pr_title, review):
    import sqlite3
    conn = sqlite3.connect("reviews.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            pr_number INTEGER,
            pr_title TEXT,
            review TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "INSERT INTO reviews (repo, pr_number, pr_title, review) VALUES (?, ?, ?, ?)",
        (repo_name, pr_number, pr_title, review)
    )
    conn.commit()
    conn.close()

def post_comment(repo_name, pr_number, review):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(f"## 🤖 AI PR Review\n\n{review}\n\n---\n*Reviewed by AI PR Reviewer*")

if __name__ == "__main__":
    repo_name = os.getenv("REPO_NAME")
    pr_number = int(os.getenv("PR_NUMBER"))
    pr_title = os.getenv("PR_TITLE", "")
    print(f"Reviewing PR #{pr_number} in {repo_name}")
    diff = get_pr_diff(repo_name, pr_number)
    review = review_diff(diff)
    save_review(repo_name, pr_number, diff["title"], review)
    post_comment(repo_name, pr_number, review)
    print("Review posted successfully")
    print("\n" + "="*50)
    print(review)
    print("="*50)