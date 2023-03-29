---
layout: "post"
show-avatar: false
title: "Git command cheat sheet"
date: "2023-03-28"
tags: [Git, cheatsheet]
cover-img: /assets/img/git_header.png
thumbnail-img: /assets/img/git_icon.png
---

credit: [reference website](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)

<img src='../../assets/img/git.png' alt="git development workflow" height="700">

- [Config your git with github](#config-your-git-with-github)
- [Local operations](#local-operations)
  - [Basic local life cycle](#basic-local-life-cycle)
  - [Tracking changes](#tracking-changes)
  - [undoing changes](#undoing-changes)
  - [Clean up files](#clean-up-files)
- [Branch management](#branch-management)
- [Remote operations](#remote-operations)
- [Create shortcut](#create-shortcut)


## Config your git with github
```bash
## config your email
git config user.email [email]

## config your username
git config user.name [username]

## set global username (email)
git config user.name --global [username]

## set global username and email by edit ~/.gitconfig
vim ~/.gitconfig
[user]
    email=[email address]
    name = [username]

## work with github personal access token
vim ~/.netrc
## past this line (change username and personal access token accordingly)
machine github.com login [username] password [personal access token]
```

## Local operations

### Basic local life cycle
```bash
## initialize local git repo
git init


## save changes (file, dir, or file match the reg expression) and add them to staging area 
git add [file/dir/reg exp]

## save all changes under current dir
git add *

## save
git add -p

## commit changes to local repo (commit all)
git commit -a -m "SOME COMMENTS"

```

### Tracking changes

```bash
## show changes that have been made since last commit
git diff

## show changes that have been made (for better granularity)
git diff --color-words

## show changes compare to one commit
git diff [HEAD/commit id] [dir/file]

## compare files from different branches
git diff [branch1] [branch2] [file]

## print all log information
git log [--oneline compact layout]

## print logs with highlights
git log --pretty

## show local repo status
git status

## git add tag to a commit
git tag -a [tagname]

## trace back to the commit history and find who made the changes
git blame [-e show email] [-w ignore wihitespace changes] [file]

## show a complete operation logs
git reflog [--relative-date show related date]
```

### undoing changes
```bash
## create a new commit with the reverse of the last commit(not really undo the commit)
git revert HEAD

## real undo the commit (recommand to use it locally, for remote use git revert)
git reset --hard [commit id]

## edit the last commit
git commit --amend

## change multiple commits
git rebase [-i interactive mode]
```

### Clean up files
```bash
## show which file will be impacted
git clean -n

## clean untracked files
git clean -f 

## clean untracked file in a dir
git clean -f -d [path]

## clean untracked and ingnored files
git clearn -x -f

## clean file in interactive mode
git clean -d -i

## remove files (reverse of git add)
git rm [-n same as what is in git clean]  [--cached remove staging index]

## undo git rm
git reset HEAD

## soft version of undo git rm
git checkout .
```

## Branch management

```bash
## show all branches of current connections (same as git branch --list)
git branch

## show all remote branches
git branch -r

## show all remote branches (same as git branch -r)
git branch -a

## create a new branch
git branch [new branch name]

## remove a branch (locally)
git branch -D [branch name]

## rename the current branch
git branch -m [branch]

## create a new branch and swithch to the branch
git checkout -b [branch name]

## remove a branch (remotely)
git push origin :branch_name

```

## Remote operations
```bash
## show remote connections
git remote [-v show url as well]

## add remote connection
git remote add [name] [url]

## remove remote connection
git remote rm [name]

## rename remote connection
git remote rename [oldname] [newname]

## inspecting a remote connection
git remote show [connection name]

## push local updates to remote  (--forece option is required after git commit --amend)
git push [remote name] [branch name]

## retrive updates from remote connection of all branches
git fetch [remote name]

## retrive updates from remote connection of certain branch
git fetch [remote name] [branch name]

## retrive all updates from all connections and all branch
git fetch --all

## merge updates from remote connection "origin/new_branch" to main
git fetch origin new_branch
git checkout main
git merge origin/new_branch

## a simpler solution to fetch+merge
git pull [remote name]

## pull without create a new commit
git pull --no-commit [remote name]

## pull with rebase (cleaner and promote linear history)
git pull --rebase [remote name]


```

## Create shortcut
```bash

## remove currently added files from staging area
git config --global alias.unstage 'reset HEAD --'
## use case: equvalent to git reset HEAD -- file
git unstage [file]
```

