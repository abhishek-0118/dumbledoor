import os
import logging
from typing import Optional, Dict, List
from ..repos.github import authenticate_github, get_github_user_info, list_github_repositories

logger = logging.getLogger("app.auth.github")


class GitHubAuthManager:
    def __init__(self, token_env: str = "GITHUB_TOKEN"):
        self.token_env = token_env
        self._token = None
        self._user_info = None
        self._authenticated = False
    
    def authenticate(self, required: bool = False) -> bool:
        try:
            token, is_authenticated = authenticate_github(self.token_env, required)
            self._token = token
            self._authenticated = is_authenticated
            
            if is_authenticated:
                self._user_info = get_github_user_info(token)
                logger.info(f"Authenticated as GitHub user: {self._user_info.get('login', 'unknown')}")
            
            return is_authenticated
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            if required:
                raise
            return False
    
    @property
    def is_authenticated(self) -> bool:
        return self._authenticated
    
    @property
    def user_info(self) -> Optional[Dict]:
        return self._user_info
    
    @property
    def token(self) -> Optional[str]:
        return self._token
    
    def list_user_repositories(self, user: Optional[str] = None) -> List[Dict]:
        if not self._authenticated:
            logger.error("Not authenticated with GitHub")
            return []
        
        return list_github_repositories(self._token, user=user)
    
    def list_org_repositories(self, org: str) -> List[Dict]:
        if not self._authenticated:
            logger.error("Not authenticated with GitHub")
            return []
        
        return list_github_repositories(self._token, org=org)
    
    def get_repository_urls(self, repositories: List[Dict]) -> List[str]:
        urls = []
        for repo in repositories:
            if repo.get("clone_url"):
                urls.append(repo["clone_url"])
            elif repo.get("html_url"):
                html_url = repo["html_url"]
                if html_url.endswith(".git"):
                    urls.append(html_url)
                else:
                    urls.append(f"{html_url}.git")
        
        return urls
    
    def get_recommended_repositories(self, limit: int = 10) -> List[str]:
        if not self._authenticated:
            logger.warning("Not authenticated - cannot get recommendations")
            return []
        
        user_repos = self.list_user_repositories()
        
        filtered_repos = []
        for repo in user_repos:
            if repo.get("fork") or repo.get("archived"):
                continue
            
            filtered_repos.append(repo)
        
        filtered_repos.sort(
            key=lambda r: (r.get("stargazers_count", 0), r.get("updated_at", "")),
            reverse=True
        )
        
        return self.get_repository_urls(filtered_repos[:limit])


def setup_github_auth(token_env: str = "GITHUB_TOKEN", required: bool = False) -> GitHubAuthManager:
    auth_manager = GitHubAuthManager(token_env)
    auth_manager.authenticate(required)
    return auth_manager
