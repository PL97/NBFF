---
layout: "post"
show-avatar: false
title: "Git command cheat sheet"
date: "2023-03-23"
tags: [Git, cheatsheet]
cover-img: /assets/img/linux.jpg
thumbnail-img: /assets/img/linux_header.jpg
---

<img src='../../assets/img/git.png' alt="git development workflow" width="600" height="128">


## Local operations
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


