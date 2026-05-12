import os
import sqlite3
import subprocess
import tempfile
from github import Github
from groq import Groq
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─── Embedding Model ──────────────────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# ─── Tool Calls ───────────────────────────────────────────────────────────────

def run_flake8(files: dict) -> str:
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in files.items():
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(tmpdir, os.path.basename(filename))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            result = subprocess.run(
                ["flake8", "--max-line-length=100", filepath],
                capture_output=True, text=True
            )
            if result.stdout:
                results.append(result.stdout.replace(filepath, filename))
    return "\n".join(results) if results else "✅ flake8: No style issues found."

def run_bandit(files: dict) -> str:
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in files.items():
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(tmpdir, os.path.basename(filename))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            result = subprocess.run(
                ["bandit", "-r", filepath, "-f", "txt", "--quiet"],
                capture_output=True, text=True
            )
            if result.stdout:
                results.append(result.stdout.replace(filepath, filename))
    return "\n".join(results) if results else "✅ bandit: No security issues found."

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
    added_contents = {}
    for f in pr.get_files():
        parsed = parse_diff(f.patch or "")
        added_contents[f.filename] = "\n".join(parsed["added"])
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
        "files": files,
        "added_contents": added_contents
    }

# ─── RAG ─────────────────────────────────────────────────────────────────────

def index_repo(repo_name: str, changed_filenames: list) -> chromadb.Collection:
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    collection = chroma_client.get_or_create_collection("repo_context")
    priority_files = ["README.md", "CONTRIBUTING.md", ".github/CONTRIBUTING.md"]
    contents = list(repo.get_contents(""))
    all_files = []
    while contents:
        item = contents.pop(0)
        if item.type == "dir":
            try:
                contents.extend(repo.get_contents(item.path))
            except:
                pass
        else:
            all_files.append(item.path)
    changed_dirs = set(os.path.dirname(f) for f in changed_filenames)
    relevant = set(priority_files)
    for f in all_files:
        if os.path.dirname(f) in changed_dirs:
            relevant.add(f)
    for filepath in relevant:
        try:
            file_content = repo.get_contents(filepath)
            text = file_content.decoded_content.decode("utf-8", errors="ignore")[:3000]
            embedding = embedder.encode(text).tolist()
            collection.upsert(
                ids=[filepath],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"filepath": filepath}]
            )
        except:
            pass
    return collection

def retrieve_context(collection: chromadb.Collection, query: str, n=3) -> str:
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n)
    context = ""
    for i, doc in enumerate(results["documents"][0]):
        filepath = results["metadatas"][0][i]["filepath"]
        context += f"\n--- {filepath} ---\n{doc}\n"
    return context

# ─── Agents ───────────────────────────────────────────────────────────────────

def build_diff_summary(diff: dict) -> str:
    summary = ""
    for f in diff["files"]:
        summary += f"\n### {f['filename']} ({f['status']})\n"
        if f["added_lines"]:
            summary += "ADDED:\n" + "\n".join(f["added_lines"]) + "\n"
        if f["removed_lines"]:
            summary += "REMOVED:\n" + "\n".join(f["removed_lines"]) + "\n"
    return summary

def run_style_agent(diff_summary, pr_title, context, flake8_output) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a code style reviewer.
You have real flake8 output. Use it as your primary source.
Format each issue as: 🔵 NITPICK: <file>:<line> - <issue>
If nothing to report, say '✅ No style issues found.'"""},
            {"role": "user", "content": f"PR: '{pr_title}'\n\nflake8:\n{flake8_output}\n\nContext:\n{context}\n\nChanges:\n{diff_summary}"}
        ]
    )
    return r.choices[0].message.content

def run_security_agent(diff_summary, pr_title, context, bandit_output) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a security reviewer.
You have real bandit scanner output. Use it as your primary source.
Format each issue as: 🔴 CRITICAL: <file>:<line> - <issue>
If nothing to report, say '✅ No security issues found.'"""},
            {"role": "user", "content": f"PR: '{pr_title}'\n\nbandit:\n{bandit_output}\n\nContext:\n{context}\n\nChanges:\n{diff_summary}"}
        ]
    )
    return r.choices[0].message.content

def run_logic_agent(diff_summary, pr_title, context) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a logic and bug reviewer.
Only review for bugs, edge cases, incorrect logic, null pointer risks, missing error handling.
Format each issue as: 🔴 CRITICAL: <file>:<line> - <issue>
If nothing to report, say '✅ No logic issues found.'"""},
            {"role": "user", "content": f"PR: '{pr_title}'\n\nContext:\n{context}\n\nChanges:\n{diff_summary}"}
        ]
    )
    return r.choices[0].message.content

def run_performance_agent(diff_summary, pr_title, context) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a performance reviewer.
Only review for inefficient algorithms, unnecessary loops, repeated DB calls, memory leaks.
Format each issue as: 🟡 SUGGESTION: <file>:<line> - <issue>
If nothing to report, say '✅ No performance issues found.'"""},
            {"role": "user", "content": f"PR: '{pr_title}'\n\nContext:\n{context}\n\nChanges:\n{diff_summary}"}
        ]
    )
    return r.choices[0].message.content

def run_coordinator_agent(agent_outputs: dict, pr_title: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    all_reviews = "\n\n".join(
        f"=== {name} ===\n{output}"
        for name, output in agent_outputs.items()
    )
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a senior engineering lead coordinating a PR review.
You have received reports from 4 specialized agents: Style, Logic, Performance, and Security.
Your job is to:
1. Synthesize the most important findings across all agents
2. Identify any conflicts or overlapping concerns between agents
3. Give a final merge recommendation

End your response with one of these three verdicts on its own line:
✅ APPROVE - Safe to merge
⚠️ NEEDS CHANGES - Minor issues to fix before merging
🚫 BLOCK - Critical issues must be resolved before merging

Be concise. Max 200 words."""},
            {"role": "user", "content": f"PR: '{pr_title}'\n\nAgent Reports:\n{all_reviews}"}
        ]
    )
    return r.choices[0].message.content

# ─── Orchestrator ─────────────────────────────────────────────────────────────

def review_diff(diff: dict, repo_name: str) -> tuple:
    diff_summary = build_diff_summary(diff)
    pr_title = diff["title"]
    changed_filenames = [f["filename"] for f in diff["files"]]
    added_contents = diff.get("added_contents", {})

    print("Running real tool checks...")
    flake8_output = run_flake8(added_contents)
    bandit_output = run_bandit(added_contents)

    print("Indexing repo for RAG context...")
    collection = index_repo(repo_name, changed_filenames)
    context = retrieve_context(collection, diff_summary[:500])

    print("Running 4 agents in parallel...")
    agent_outputs = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_style_agent, diff_summary, pr_title, context, flake8_output): "🎨 Style & Readability",
            executor.submit(run_security_agent, diff_summary, pr_title, context, bandit_output): "🔒 Security",
            executor.submit(run_logic_agent, diff_summary, pr_title, context): "🐛 Logic & Bugs",
            executor.submit(run_performance_agent, diff_summary, pr_title, context): "⚡ Performance",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                agent_outputs[name] = future.result()
            except Exception as e:
                agent_outputs[name] = f"❌ Agent failed: {str(e)}"

    print("Running coordinator agent...")
    coordinator_output = run_coordinator_agent(agent_outputs, pr_title)

    order = ["🎨 Style & Readability", "🐛 Logic & Bugs", "⚡ Performance", "🔒 Security"]
    detailed_review = "\n\n---\n\n".join(
        f"### {name}\n\n{agent_outputs[name]}"
        for name in order if name in agent_outputs
    )
    return detailed_review, coordinator_output

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

def post_comment(repo_name, pr_number, review, coordinator, score):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    score_emoji = "🟢" if score["risk_score"] >= 80 else "🟡" if score["risk_score"] >= 50 else "🔴"
    comment = f"""## 🤖 AI PR Review

{score_emoji} **Risk Score: {score['risk_score']}/100** &nbsp;|&nbsp; 🔴 {score['critical']} Critical &nbsp;|&nbsp; 🟡 {score['suggestions']} Suggestions &nbsp;|&nbsp; 🔵 {score['nitpicks']} Nitpicks

---

## 🧠 Coordinator Summary

{coordinator}

---

## 📋 Detailed Agent Reports

{review}

---
*Reviewed by AI PR Reviewer · LLaMA 3.3 + flake8 + bandit*"""
    pr.create_issue_comment(comment)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo_name = os.getenv("REPO_NAME")
    pr_number = int(os.getenv("PR_NUMBER"))
    print(f"Reviewing PR #{pr_number} in {repo_name}")
    diff = get_pr_diff(repo_name, pr_number)
    review, coordinator = review_diff(diff, repo_name)
    score = compute_score(review)
    save_review(repo_name, pr_number, diff["title"], review, score)
    post_comment(repo_name, pr_number, review, coordinator, score)
    print(f"Review posted — Risk Score: {score['risk_score']}/100")