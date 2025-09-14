"""Technology keywords for query analysis and detection"""

# Technology keywords for query analysis
TECH_KEYWORDS = [
    "python", "javascript", "typescript", "react", "node", "go", "java", 
    "rust", "docker", "kubernetes", "api", "database", "sql", "mongodb",
    "fastapi", "flask", "django", "express", "vue", "angular", "nextjs",
    "redis", "postgres", "mysql", "elasticsearch", "kafka", "rabbitmq",
    "aws", "azure", "gcp", "terraform", "jenkins", "github", "gitlab",
    "microservices", "graphql", "rest", "grpc", "websocket", "oauth",
    "jwt", "authentication", "authorization", "security", "encryption"
]

# Query analysis keyword patterns
QUERY_ANALYSIS_KEYWORDS = {
    "how_to": ["how to", "how do", "how can", "how should"],
    "what_is": ["what is", "what does", "what are"],
    "debugging": ["error", "bug", "issue", "problem", "fix", "debug", "troubleshoot"],
    "implementation": ["implement", "create", "build", "develop", "code", "write"],
    "architectural": ["architecture", "structure", "design", "pattern", "overview", "flow", 
                     "interconnectivity", "cross-repo", "between repos", "system design", "high level"],
    "code_related": ["function", "class", "method", "variable", "import", "module"],
    "test_related": ["test", "testing", "unit test", "spec", "pytest", "jest"],
    "config_related": ["config", "configuration", "settings", "setup", "environment"]
}

# Language-specific keywords
LANGUAGE_KEYWORDS = {
    "python": ["python", "pip", "venv", "django", "flask", "fastapi", "pytest", "numpy", "pandas"],
    "javascript": ["javascript", "js", "npm", "yarn", "node", "react", "vue", "angular", "jest"],
    "typescript": ["typescript", "ts", "interface", "type", "generic", "decorator"],
    "go": ["go", "golang", "goroutine", "channel", "interface", "struct", "package"],
    "java": ["java", "spring", "maven", "gradle", "junit", "class", "interface", "annotation"],
    "rust": ["rust", "cargo", "trait", "struct", "enum", "impl", "match"],
    "docker": ["docker", "dockerfile", "container", "image", "compose", "volume", "network"],
    "kubernetes": ["k8s", "kubernetes", "pod", "service", "deployment", "ingress", "configmap"],
}

# Framework and library keywords
FRAMEWORK_KEYWORDS = {
    "web": ["fastapi", "flask", "django", "express", "koa", "nest", "spring", "gin"],
    "frontend": ["react", "vue", "angular", "svelte", "nextjs", "nuxt", "gatsby"],
    "testing": ["pytest", "jest", "mocha", "jasmine", "junit", "testng", "rspec"],
    "database": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite"],
    "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "helm"],
}

# File pattern keywords
FILE_PATTERN_KEYWORDS = {
    "test_patterns": ["test", "spec", "__test__", "__spec__"],
    "config_patterns": ["config", "conf", "settings", "env"],
    "build_patterns": ["build", "dist", "target", "bin", "output"],
    "docs_patterns": ["docs", "documentation", "readme", "changelog"],
}

# Architecture pattern keywords
ARCHITECTURE_KEYWORDS = [
    "microservices", "monolith", "serverless", "event-driven", "pub-sub",
    "mvc", "mvp", "mvvm", "clean architecture", "hexagonal", "onion",
    "repository pattern", "factory pattern", "singleton", "observer",
    "middleware", "interceptor", "decorator", "adapter", "facade"
]

# Performance and optimization keywords
PERFORMANCE_KEYWORDS = [
    "performance", "optimization", "caching", "indexing", "pagination",
    "lazy loading", "connection pooling", "batch processing", "async",
    "concurrency", "parallelism", "threading", "multiprocessing"
]
