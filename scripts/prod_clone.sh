#!/usr/bin/env bash
set -euo pipefail

echo "[prod_clone] Starting repository cloning process"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "[prod_clone] WARNING: GITHUB_TOKEN not set. Public repositories only."
else
    echo "[prod_clone] GitHub token detected, validating..."
    if ! curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user > /dev/null; then
        echo "[prod_clone] ERROR: Invalid GitHub token"
        exit 1
    fi
    echo "[prod_clone] GitHub token validated successfully"
fi

python - <<'PY'
import os
import sys
import logging
from pathlib import Path
from app.config.loader import load_config
from app.auth import setup_github_auth
from app.repos.github import clone_or_pull

logging.basicConfig(level=logging.INFO, format='[prod_clone] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        cfg = load_config("prod")
        repo_urls = cfg.indexing.repo_urls or []
        
        if not repo_urls:
            logger.warning("No repository URLs configured in prod.yaml")
            return 1
        
        logger.info(f"Found {len(repo_urls)} repositories to clone")
        
        if cfg.indexing.github_auth_required:
            try:
                auth_manager = setup_github_auth(
                    cfg.indexing.github_token_env,
                    True
                )
                logger.info(f"Authenticated with GitHub as: {auth_manager.user_info.get('login', 'unknown')}")
            except Exception as e:
                logger.error(f"GitHub authentication failed: {e}")
                return 1
        
        root_dir = Path(cfg.indexing.local_repo_root).resolve()
        root_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Repository root: {root_dir}")
        
        success_count = 0
        failed_repos = []
        
        for url in repo_urls:
            url = url.strip()
            if not url:
                continue
                
            try:
                logger.info(f"Processing repository: {url}")
                repo_dir = clone_or_pull(
                    url,
                    root_dir,
                    cfg.indexing.github_token_env,
                    cfg.indexing.github_auth_required
                )
                logger.info(f"✓ Successfully processed: {repo_dir.name}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to process {url}: {e}")
                failed_repos.append(url)
                continue
        
        logger.info(f"Cloning complete: {success_count}/{len(repo_urls)} repositories processed successfully")
        
        if failed_repos:
            logger.warning(f"Failed repositories:")
            for repo in failed_repos:
                logger.warning(f"  - {repo}")
        
        if success_count == 0:
            logger.error("No repositories were cloned successfully")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
PY

echo "[prod_clone] Repository cloning process completed"
