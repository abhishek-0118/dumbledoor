import argparse
import logging
from pathlib import Path
from .config.loader import load_config
from .indexing.indexer import RepoIndexer
from .api.server import app
from .auth import setup_github_auth
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app.cli")

def main():
	ap = argparse.ArgumentParser(description="Repository Indexing and Search Service")
	ap.add_argument("--index", action="store_true", help="Index repositories")
	ap.add_argument("--serve", action="store_true", help="Start the API server")
	ap.add_argument("--env", type=str, default=None, help="Override APP_ENV (local|prod)")
	ap.add_argument("--auth-test", action="store_true", help="Test GitHub authentication")
	ap.add_argument("--list-repos", action="store_true", help="List available repositories")
	ap.add_argument("--force-cleanup", action="store_true", help="Remove all embeddings and indexes")
	ap.add_argument("--force-index", action="store_true", help="Clean up and re-index all repositories")
	ap.add_argument("--test-search", type=str, help="Test search functionality with a query")
	args = ap.parse_args()

	try:
		cfg = load_config(args.env)
		logger.info(f"Loaded configuration for environment: {cfg.app_env}")
	except Exception as e:
		logger.error(f"Failed to load configuration: {e}")
		return 1

	if args.auth_test:
		try:
			auth_manager = setup_github_auth(
				cfg.indexing.github_token_env,
				cfg.indexing.github_auth_required
			)
			if auth_manager.is_authenticated:
				user_info = auth_manager.user_info
				logger.info(f"✓ GitHub authentication successful")
				logger.info(f"  User: {user_info.get('login', 'unknown')}")
				logger.info(f"  Name: {user_info.get('name', 'N/A')}")
				logger.info(f"  Email: {user_info.get('email', 'N/A')}")
			else:
				logger.warning("GitHub authentication failed or not configured")
		except Exception as e:
			logger.error(f"GitHub authentication error: {e}")
		return 0

	if args.list_repos:
		try:
			auth_manager = setup_github_auth(
				cfg.indexing.github_token_env,
				False
			)
			if auth_manager.is_authenticated:
				repos = auth_manager.list_user_repositories()
				logger.info(f"Found {len(repos)} repositories:")
				for repo in repos[:10]:
					logger.info(f"  - {repo['full_name']} ({repo.get('stargazers_count', 0)})")
				if len(repos) > 10:
					logger.info(f"  ... and {len(repos) - 10} more")
			else:
				logger.warning("GitHub authentication required to list repositories")
		except Exception as e:
			logger.error(f"Error listing repositories: {e}")
		return 0

	try:
		idx = RepoIndexer(cfg)
		logger.info("Repository indexer initialized")
	except Exception as e:
		logger.error(f"Failed to initialize indexer: {e}")
		return 1

	if args.force_cleanup:
		try:
			logger.info("Starting force cleanup - removing all embeddings and indexes")
			idx.cleanup_all()
			logger.info("Force cleanup completed successfully")
			return 0
		except Exception as e:
			logger.error(f"Force cleanup failed: {e}")
			return 1

	if args.force_index:
		try:
			logger.info("Starting force index - cleaning up and re-indexing all repositories")
			idx.cleanup_all()
			logger.info("Cleanup completed, starting re-indexing...")
			
			if cfg.app_env == "local":
				root = Path(cfg.indexing.local_repo_root).resolve()
				logger.info(f"Re-indexing repositories from: {root}")
				
				if not root.exists() or not any(root.iterdir()):
					logger.warning(f"No repositories found in {root}")
					if cfg.indexing.local_paths:
						logger.info("Re-indexing configured local paths instead")
						idx.index_local_paths(cfg.indexing.local_paths)
					else:
						logger.warning("No local paths configured for indexing")
				else:
					idx.index_local_root(root)
			else:
				if not cfg.indexing.repo_urls:
					logger.error("No repo_urls configured for prod. Set indexing.repo_urls in config.")
					return 1
				
				if cfg.indexing.github_auth_required:
					auth_manager = setup_github_auth(
						cfg.indexing.github_token_env,
						True
					)
					logger.info(f"Authenticated with GitHub as: {auth_manager.user_info.get('login', 'unknown')}")
				
				logger.info(f"Cloning and re-indexing {len(cfg.indexing.repo_urls)} repository URLs")
				idx.index_repo_urls(cfg.indexing.repo_urls)
			
			logger.info("Force index completed successfully")
			return 0
		except Exception as e:
			logger.error(f"Force index failed: {e}")
			return 1

	if args.test_search:
		try:
			from .search.retrieval import enhanced_search, create_context_summary
			logger.info(f"Testing search with query: '{args.test_search}'")
			
			docs = enhanced_search(idx.store, args.test_search, cfg, k=10)
			context_summary = create_context_summary(docs)
			
			logger.info(f"Found {len(docs)} relevant documents")
			logger.info(f"Context summary: {context_summary}")
			
			for i, doc in enumerate(docs[:5]):
				metadata = doc.metadata or {}
				logger.info(f"\n--- Result {i+1} ---")
				logger.info(f"File: {metadata.get('repo', 'unknown')}/{metadata.get('path', 'unknown')}")
				logger.info(f"Language: {metadata.get('language', 'unknown')}")
				logger.info(f"Module: {metadata.get('module_name', 'N/A')}")
				logger.info(f"Preview: {doc.page_content[:200]}...")
			
			return 0
		except Exception as e:
			logger.error(f"Search test failed: {e}")
			return 1

	if args.index:
		try:
			if cfg.app_env == "local":
				root = Path(cfg.indexing.local_repo_root).resolve()
				logger.info(f"Indexing repositories from: {root}")
				
				if not root.exists() or not any(root.iterdir()):
					logger.warning(f"No repositories found in {root}")
					if cfg.indexing.local_paths:
						logger.info("Indexing configured local paths instead")
						idx.index_local_paths(cfg.indexing.local_paths)
					else:
						logger.warning("No local paths configured for indexing")
				else:
					idx.index_local_root(root)
				return 0
			else:
				if not cfg.indexing.repo_urls:
					logger.error("No repo_urls configured for prod. Set indexing.repo_urls in config.")
					return 1
				
				if cfg.indexing.github_auth_required:
					auth_manager = setup_github_auth(
						cfg.indexing.github_token_env,
						True
					)
					logger.info(f"Authenticated with GitHub as: {auth_manager.user_info.get('login', 'unknown')}")
				
				logger.info(f"Cloning and indexing {len(cfg.indexing.repo_urls)} repository URLs")
				idx.index_repo_urls(cfg.indexing.repo_urls)
				return 0
		except Exception as e:
			logger.error(f"Indexing failed: {e}")
			return 1
	
	if args.serve:
		try:
			logger.info(f"Starting server on {cfg.server.host}:{cfg.server.port}")
			uvicorn.run(
				app, 
				host=cfg.server.host, 
				port=cfg.server.port,
				log_level="info"
			)
			return 0
		except Exception as e:
			logger.error(f"Server failed to start: {e}")
			return 1
	
	# If no action specified, show help
	if not any([args.index, args.serve, args.auth_test, args.list_repos, args.force_cleanup, args.force_index, args.test_search]):
		ap.print_help()
		return 0
	
	return 0

if __name__ == "__main__":
	exit(main())
