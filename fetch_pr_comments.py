#!/usr/bin/env python3
"""
Fetch PR review comments from GitHub for multiple PRs and save as CSV.
Handles pagination and both issue-level and inline review comments.
"""

import urllib.request
import urllib.error
import json
import csv
import time

# Configuration
OWNER = "alibaba"
REPO = "rtp-llm"
REVIEWER = "LLLLKKKK"
PR_LIST = [1053, 1071, 1075, 1096]
OUTPUT_FILE = "/workspace/rtp-llm-xpu/pr_review_comments.csv"

def fetch_comments(url, headers):
    """Fetch all pages of comments from a given URL."""
    all_comments = []
    page = 1
    
    while True:
        paginated_url = f"{url}&page={page}"
        print(f"  Fetching page {page}...")
        
        try:
            req = urllib.request.Request(paginated_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"  HTTP Error {e.code}: {e.reason}")
            break
        except Exception as e:
            print(f"  Error fetching page: {e}")
            break
        
        if not data or len(data) == 0:
            break
            
        all_comments.extend(data)
        print(f"    Got {len(data)} comments on page {page}")
        
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.5)
    
    return all_comments

def extract_category(body):
    """Extract P-level category from comment body."""
    body_upper = body.upper()
    if "[P0]" in body_upper or "P0:" in body_upper:
        return "P0"
    elif "[P1]" in body_upper or "P1:" in body_upper:
        return "P1"
    elif "[P2]" in body_upper or "P2:" in body_upper:
        return "P2"
    elif "[P3]" in body_upper or "P3:" in body_upper:
        return "P3"
    return "P3"

def main():
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-PR-Fetch-Script"
    }
    
    all_issues = []
    issue_id = 1
    
    for pr_num in PR_LIST:
        print(f"\n{"="*60}")
        print(f"Processing PR #{pr_num}")
        print(f"{"="*60}")
        
        # 1. Fetch issue-level comments
        issue_comments_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{pr_num}/comments?per_page=100"
        print(f"\nFetching issue-level comments for PR #{pr_num}...")
        
        issue_comments = fetch_comments(issue_comments_url, headers)
        filtered_issue = [c for c in issue_comments if c.get("user", {}).get("login") == REVIEWER]
        print(f"  Found {len(filtered_issue)} comments from {REVIEWER} in issue comments")
        
        for c in filtered_issue:
            category = extract_category(c["body"])
            issue_summary = c["body"]
            if len(issue_summary) > 5000:
                issue_summary = issue_summary[:5000] + "...[truncated]"
            
            all_issues.append({
                "Issue ID": issue_id,
                "Issue Category": category,
                "PR": f"#{pr_num}",
                "Issue Summary": issue_summary,
                "Issue Suggestion": "",
                "Commented Time": c["created_at"]
            })
            issue_id += 1
        
        # 2. Fetch inline review comments
        inline_comments_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_num}/comments?per_page=100"
        print(f"\nFetching inline review comments for PR #{pr_num}...")
        
        inline_comments = fetch_comments(inline_comments_url, headers)
        filtered_inline = sorted(
            [c for c in inline_comments if c.get("user", {}).get("login") == REVIEWER],
            key=lambda x: x["created_at"]
        )
        print(f"  Found {len(filtered_inline)} inline comments from {REVIEWER}")
        
        for c in filtered_inline:
            category = extract_category(c["body"])
            path = c.get("path", "unknown")
            line = c.get("line", c.get("original_line", "unknown"))
            issue_summary = f"[File: {path}:{line}] {c["body"]}"
            
            if len(issue_summary) > 5000:
                issue_summary = issue_summary[:5000] + "...[truncated]"
            
            all_issues.append({
                "Issue ID": issue_id,
                "Issue Category": category,
                "PR": f"#{pr_num}",
                "Issue Summary": issue_summary,
                "Issue Suggestion": "",
                "Commented Time": c["created_at"]
            })
            issue_id += 1
        
        time.sleep(1)
    
    # Write to CSV
    print(f"\n{"="*60}")
    print(f"Writing {len(all_issues)} issues to {OUTPUT_FILE}")
    print(f"{"="*60}")
    
    fieldnames = ["Issue ID", "Issue Category", "PR", "Issue Summary", "Issue Suggestion", "Commented Time"]
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for issue in all_issues:
            writer.writerow(issue)
    
    print(f"\nDone! Saved {len(all_issues)} review comments to {OUTPUT_FILE}")
    
    # Print summary by PR and category
    print(f"\n{"="*60}")
    print("Summary by PR:")
    print(f"{"="*60}")
    for pr_num in PR_LIST:
        pr_issues = [i for i in all_issues if i["PR"] == f"#{pr_num}"]
        p0 = len([i for i in pr_issues if i["Issue Category"] == "P0"])
        p1 = len([i for i in pr_issues if i["Issue Category"] == "P1"])
        p2 = len([i for i in pr_issues if i["Issue Category"] == "P2"])
        p3 = len([i for i in pr_issues if i["Issue Category"] == "P3"])
        print(f"PR #{pr_num}: Total={len(pr_issues)} (P0={p0}, P1={p1}, P2={p2}, P3={p3})")

if __name__ == "__main__":
    main()
