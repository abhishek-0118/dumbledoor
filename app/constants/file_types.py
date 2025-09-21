"""File type constants and mappings"""

# Code file extensions that should be indexed
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.java', '.kt', '.rs', 
    '.c', '.cpp', '.h', '.hpp', '.rb', '.php', '.scala', '.sql', 
    '.sh', '.md', '.yml', '.yaml', '.toml', '.ini', '.json', '.xml', 
    '.html', '.css'
}

# Language detection mapping
LANGUAGE_MAP = {
    '.py': 'python',
    '.js': 'javascript', 
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.jsx': 'javascript',
    '.go': 'go',
    '.java': 'java',
    '.kt': 'kotlin',
    '.rs': 'rust',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.rb': 'ruby',
    '.php': 'php',
    '.scala': 'scala',
    '.sql': 'sql',
    '.sh': 'bash',
    '.md': 'markdown',
    '.yml': 'yaml',
    '.yaml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.json': 'json',
    '.xml': 'xml',
    '.html': 'html',
    '.css': 'css',
}

# Test file detection patterns
TEST_FILE_INDICATORS = [
    'test', 'spec', '__test__', '__spec__', 'tests/', 'spec/',
    '.test.', '.spec.', '_test.', '_spec.'
]

# Configuration file detection patterns
CONFIG_FILE_INDICATORS = [
    'config', 'conf', 'settings', 'setup', 'makefile', 'dockerfile',
    '.env', '.ini', '.toml', '.yaml', '.yml', '.json', 'package.json',
    'requirements.txt', 'go.mod', 'cargo.toml'
]

