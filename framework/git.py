import subprocess


def get_commit_hash():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

def get_git_commit_summary():
    short_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True) \
        .stdout.strip()

    message = subprocess.run(["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True) \
       .stdout.strip()

    return f'{short_hash}_{message}'
