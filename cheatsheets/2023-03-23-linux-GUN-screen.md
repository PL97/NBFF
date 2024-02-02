---
layout: "post"
show-avatar: false
title: "linux GUN screen command cheat sheet"
date: "2023-03-23"
tags: [linux, cheatsheet]
cover-img: /assets/img/linux.jpg
thumbnail-img: /assets/img/linux_header.jpg
---

credit to this great [blog](https://linuxize.com/post/how-to-use-linux-screen/)



## GUN screen
```bash
# to start a new screen session
screen

# start a new screen session with specified name
screen -S [session name]

# display active screen sessions
screen -ls

# resume a detached screen
screen -r/-x [session id]



``` 

Basic command with screen

```bash
# display avaliable command (can find reference for belows)
ctrl+a ?

# create a new window (with shell)
ctrl+a c

# list all windows
ctrl+a \"

# switch to window by number
ctrl+a [number]

# rename the crrent window
ctrl+a A

# split the current region horizontally
ctrl+a S

# split the current region vertically (combine with ctrl+a tab and ctrl+a c)
ctrl+a |

# swith to the next region in the current window
ctrl+a tab

# create a new window (with shell)
ctrl+a c

# toggle between current and previous window
ctrl+a ctrl+a

# close all regions but the current one
ctrl+a Q

# close the current region
ctrl+a X

# detach from linux screen session
ctrl+a ctrl+d

# close screen session
ctrl+d

# move to next screen window
ctrl+a n

# move to previous screen window
ctrl+a n

# freeze screen
ctrl+a ctrl+s

# unfreeze screen
ctrl+a ctrl+q
```

:star: customize your screen session!

edit ~/.screenrc

A example:

```bash
# Turn off the welcome message
startup_message off

# Disable visual bell
vbell off

# Set scrollback buffer to 10000
defscrollback 10000

# Customize the status line
hardstatus alwayslastline
hardstatus string '%{= kG}[ %{G}%H %{g}][%= %{= kw}%?%-Lw%?%{r}(%{W}%n*%f%t%?(%u)%?%{r})%{w}%?%+Lw%?%?%= %{g}][%{B} %m-%d %{W}%c %{g}]'

```

