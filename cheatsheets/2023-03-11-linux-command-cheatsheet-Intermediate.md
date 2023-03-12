---
layout: "post"
show-avatar: false
title: "Bash command cheat sheet (Intermediate)"
date: "2023-03-11"
tags: [linux, cheatsheet]
cover-img: /assets/img/linux.jpg
thumbnail-img: /assets/img/linux_header.jpg
---


- [Bash Script Template](#bash-script-template)
- [User and Groups](#user-and-groups)
- [Logical Operator](#logical-operator)
- [Pip \& redirect](#pip--redirect)
- [Arguments](#arguments)





### Bash Script Template
```bash
#!/usr/bin/env bash

[your command here]
```

### User and Groups
```bash
# change owner of a file or dir
chown [owner] [file]

# change pprivileges
chmod [-r] [+/-rwn] [dir]

# create a new user
adduser [user]

# list all user
cat /etc/passwd

# delete user
deluser [user]

# print groups
groups

# create a new group
groupadd

# change group privileges
groupmod

# delete group
delgroup

# change password
passwd
```

### Logical Operator
```bash
# and, execture all otherwise none
[command1] && [command2]

# or, execute any if possible
[command1] || [command2]

# condition
if [ condition ]; then [command1]; else [command2]; fi
```

### Pip & redirect

```bash
# make a pip 
[command1] | [command2] | [command3]

# sort files based on the name, and display it using less
ls | grep *.sh | less

# redirect stdout to file (overwrite), reverse "<"
[command] > [file]

# same as above
[command] 1>[file]


# redirect stderr to file
[command] 2>[file]

# redicect stderr and stdout to file
[command] &>name

# redirect stdout to file (append), reverse "<<"
[command] >> [file]

# discard output
[command] > /dev/null

# discard error message, same as 2>&0
[command] 2>/dev/null

# discard output and error message
[command] > /dev/null 2>/dev/null

# same as above, different from &>, &1 indicate stdout
[command] > dev/null 2>&1
```

### Arguments

```bash
# use arg when execute bash script
[command] "${arg1}"

# same as above
[command] $1
```