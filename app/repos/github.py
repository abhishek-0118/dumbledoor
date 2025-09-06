import os
import re
import subprocess
import requests
from pathlib import Path
from typing import Optional, Tuple


def validate_github_token(token: str) -> bool:
	try:
		headers = {"Authorization": f"token {token}"}
		response = requests.get("https://api.github.com/user", headers=headers)
		return response.status_code == 200
	except Exception:
		return False


def get_github_user_info(token: str) -> dict:
	try:
		headers = {"Authorization": f"token {token}"}
		response = requests.get("https://api.github.com/user", headers=headers)
		if response.status_code == 200:
			return response.json()
		return {}
	except Exception:
		return {}


def authenticate_github(token_env: Optional[str] = "GITHUB_TOKEN", required: bool = False) -> Tuple[Optional[str], bool]:
	if not token_env:
		if required:
			raise ValueError("GitHub authentication is required but no token environment variable specified")
		return None, False
	
	token = os.getenv(token_env)
	if not token:
		if required:
			raise ValueError(f"GitHub token not found in environment variable: {token_env}")
		return None, False
	
	if not validate_github_token(token):
		if required:
			raise ValueError(f"Invalid GitHub token from environment variable: {token_env}")
		return None, False
	
	return token, True


def clone_or_pull(url: str, root: Path, token_env: Optional[str] = "GITHUB_TOKEN", auth_required: bool = False) -> Path:
	token, is_authenticated = authenticate_github(token_env, auth_required)
	
	clean_url = url.rstrip("/")
	
	if token and clean_url.startswith("https://") and "@" not in clean_url:
		clean_url = clean_url.replace("https://", f"https://{token}:x-oauth-basic@")
	
	name = re.sub(r"[^A-Za-z0-9_.-]", "-", clean_url.split("/")[-1].replace(".git", ""))
	dest = root / name
	
	try:
		if not dest.exists():
			print(f"Cloning repository: {url}")
			subprocess.check_call(["git", "clone", "--depth", "1", clean_url, str(dest)])
		else:
			print(f"Updating repository: {name}")
			subprocess.check_call(["git", "fetch", "--all", "--prune"], cwd=dest)
			subprocess.check_call(["git", "reset", "--hard", "origin/HEAD"], cwd=dest)
		
		return dest
		
	except subprocess.CalledProcessError as e:
		if auth_required and not is_authenticated:
			raise ValueError(f"Failed to clone {url}. Authentication may be required. Error: {e}")
		else:
			raise ValueError(f"Failed to clone/update repository {url}: {e}")


def list_github_repositories(token: str, user: Optional[str] = None, org: Optional[str] = None) -> list:
	try:
		headers = {"Authorization": f"token {token}"}
		
		if org:
			url = f"https://api.github.com/orgs/{org}/repos"
		elif user:
			url = f"https://api.github.com/users/{user}/repos"
		else:
			url = "https://api.github.com/user/repos"
		
		repos = []
		page = 1
		
		while True:
			response = requests.get(url, headers=headers, params={"page": page, "per_page": 100})
			if response.status_code != 200:
				break
			
			page_repos = response.json()
			if not page_repos:
				break
			
			repos.extend(page_repos)
			page += 1
		
		return repos
		
	except Exception as e:
		print(f"Error fetching repositories: {e}")
		return []
