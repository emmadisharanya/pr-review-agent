import os
import sqlite3
from github import Github
from groq import Groq
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ─── Diff Parsing ────────────────────────────────────────────────────────────

def parse_diff(patch: str) -> dict:
    added, removed, context = [], [], []
    if not patch:
        return {"added": added, "removed": removed, "context": context}
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        else:
            context.append(line)
    return {"added": added, "removed": removed, "context": context}

def get_pr_diff(repo_name, pr_number):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    files = []
    for f in pr.get_files():
        parsed = parse_diff(f.patch or "")
        files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": f.patch or "",
            "added_lines": parsed["added"],
            "removed_lines": parsed["removed"],
            "context_lines": parsed["context"]
        })
    return {
        "title": pr.title,
        "body": pr.body,
        "files": files
    }

# ─── Multi-Agent Reviewers ───────────────────────────────────────────────────

def build_diff_summary(diff: dict) -> str:
    summary = ""
    for f in diff["files"]:
        summary += f"\n### {f['filename']} ({f['status']})\n"
        if f["added_lines"]:
            summary += "ADDED:\n" + "\n".join(f["added_lines"]) + "\n"
        if f["removed_lines"]:
            summary += "REMOVED:\n" + "\n".join(f["removed_lines"]) + "\n"
    return summary

def run_agent(agent_name: str, system_prompt: str, diff_summary: str, pr_title: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"PR: '{pr_title}'\n\nChanges:\n{diff_summary}"}
        ]
    )
    return f"### {agent_name}\n\n{response.choices[0].message.content}"

AGENTS = {
    "🎨 Style & Readability": """You are a code style reviewer. Only review for:
- Naming conventions, readability, formatting
- Unnecessary complexity, dead code
Format each issue as: 🔵 NITPICK: <file>:<line> - <issue>
If nothing to report, say '✅ No style issues found.'""",

    "🐛 Logic & Bugs": """You are a logic and bug reviewer. Only review for:
- Bugs, edge cases, incorrect logic, null pointer risks
- Off-by-one errors, missing error handling
Format each issue as: 🔴 CRITICAL: <file>:<line> - <issue>
If nothing to report, say '✅ No logic issues found.'""",

    "⚡ Performance": """You are a performance reviewer. Only review for:
- Inefficient algorithms, unnecessary loops, repeated DB calls
- Memory leaks, blocking operations
Format each issue as: 🟡 SUGGESTION: <file>:<line> - <issue>
If nothing to report, say '✅ No performance issues found.'""",

    "🔒 Security": """You are a security reviewer. Only review for:
- SQL injection, XSS, hardcoded secrets, insecure dependencies
- Authentication/authorization issues, exposed sensitive data
Format each issue as: 🔴 CRITICAL: <file>:<line> - <issue>
If nothing to report, say '✅ No security issues found.'"""
}

def review_diff(diff: dict) -> str:
    diff_summary = build_diff_summary(diff)
    pr_title = diff["title"]
    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_agent, name, prompt, diff_summary, pr_title): name
            for name, prompt in AGENTS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"### {name}\n\n❌ Agent failed: {str(e)}"

    return "\n\n---\n\n".join(results[name] for name in AGENTS if name in results)

# ─── Scoring ─────────────────────────────────────────────────────────────────

def compute_score(review: str) -> dict:
    critical = review.count("🔴 CRITICAL")
    suggestions = review.count("🟡 SUGGESTION")
    nitpicks = review.count("🔵 NITPICK")
    risk_score = max(0, 100 - (critical * 20) - (suggestions * 5) - (nitpicks * 1))
    return {
        "risk_score": risk_score,
        "critical": critical,
        "suggestions": suggestions,
        "nitpicks": nitpicks
    }

# ─── Storage ──────────────────────────────────────────────────────────────────

def save_review(repo_name, pr_number, pr_title, review, score):
    conn = sqlite3.connect("reviews.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            pr_number INTEGER,
            pr_title TEXT,
            review TEXT,
            risk_score INTEGER,
            critical_count INTEGER,
            suggestion_count INTEGER,
            nitpick_count INTEGER,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "INSERT INTO reviews (repo, pr_number, pr_title, review, risk_score, critical_count, suggestion_count, nitpick_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (repo_name, pr_number, pr_title, review, score["risk_score"], score["critical"], score["suggestions"], score["nitpicks"])
    )
    conn.commit()
    conn.close()

# ─── GitHub Comment ───────────────────────────────────────────────────────────

def post_comment(repo_name, pr_number, review, score):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    score_emoji = "🟢" if score["risk_score"] >= 80 else "🟡" if score["risk_score"] >= 50 else "🔴"

    header = f"""## 🤖 AI PR Review

{score_emoji} **Risk Score: {score['risk_score']}/100** &nbsp;|&nbsp; 🔴 {score['critical']} Critical &nbsp;|&nbsp; 🟡 {score['suggestions']} Suggestions &nbsp;|&nbsp; 🔵 {score['nitpicks']} Nitpicks

---

"""
    pr.create_issue_comment(header + review + "\n\n---\n*Reviewed by AI PR Reviewer · Powered by LLaMA 3.3*")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo_name = os.getenv("REPO_NAME")
    pr_number = int(os.getenv("PR_NUMBER"))
    print(f"Reviewing PR #{pr_number} in {repo_name}")
    diff = get_pr_diff(repo_name, pr_number)
    review = review_diff(diff)
    score = compute_score(review)
    save_review(repo_name, pr_number, diff["title"], review, score)
    post_comment(repo_name, pr_number, review, score)
    print(f"Review posted — Risk Score: {score['risk_score']}/100")