import os
import json
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
                "content": "You are a senior software engineer reviewing a pull request. Be concise and specific."
            },
            {
                "role": "user",
                "content": f"Review this PR titled '{diff['title']}'.\n\nChanges:\n{files_text}\n\nPoint out bugs, security issues, and improvements. Be specific about file and line."
            }
        ]
    )
    return response.choices[0].message.content

def post_comment(repo_name, pr_number, review):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(f"## AI PR Review\n\n{review}")

if __name__ == "__main__":
    repo_name = os.getenv("REPO_NAME")
    pr_number = int(os.getenv("PR_NUMBER"))
    print(f"Reviewing PR #{pr_number} in {repo_name}")
    diff = get_pr_diff(repo_name, pr_number)
    review = review_diff(diff)
    post_comment(repo_name, pr_number, review)
    print("Review posted successfully")