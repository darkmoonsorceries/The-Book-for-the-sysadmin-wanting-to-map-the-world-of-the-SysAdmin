#!/bin/bash
# Creates a git worktree for working on the main branch simultaneously
WORKTREE_PATH="../sysadmin-worktree"

if git worktree list | grep -q "$WORKTREE_PATH"; then
    echo "Worktree already exists at $WORKTREE_PATH"
else
    git worktree add "$WORKTREE_PATH" main
    echo "Worktree created at $WORKTREE_PATH on branch main"
fi
