from github import Github
import os

def get_pr_diff(repo_name: str, pr_number: int) -> dict:
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
        "base_branch": pr.base.ref,
        "head_branch": pr.head.ref,
        "files": files
    }

def post_review_comment(repo_name: str, pr_number: int, review: str):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(f"## AI PR Review\n\n{review}")