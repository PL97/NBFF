---
layout: "post"
show-avatar: false
title: "linux command cheat sheet (Basic)"
date: "2023-03-11"
tags: [linux, cheatsheet]
cover-img: /assets/img/linux.jpg
thumbnail-img: /assets/img/linux_header.jpg
---


*This note mostly adatped from [linuxgems](https://github.com/kevinthew/linuxgems)*


- [Bash Script Template](#bash-script-template)
- [General Command](#general-command)
- [Basic File and Directory Operations](#basic-file-and-directory-operations)
- [File System Administration](#file-system-administration)
- [files operation](#files-operation)
- [Processes](#processes)
- [Compression and Encryption](#compression-and-encryption)




### Bash Script Template
```bash
#!/usr/bin/env bash

[your command here]
```

### General Command
```bash
# view thet manual for a command
man [command] || [command] -h

# guess the name of command by a vague name
apropos [guess]

# view index of help pages
info
```

### Basic File and Directory Operations
```bash
# print current working dir
pwd

# show the list of files (-a show all files including .file, 
# -l show complete file status (create date, permission, and etc.))
ls -a -l

# display files in tree structure
tree

# move files
mv [source] [target]

# rename a file
rename [pattern] files

# remove spaces from filenames in current directory
rename -n 's/[\s]/''/g' *

# change capitals to lowercase in filenames in current directory
rename 'y/A-Z/a-z/' *

# delete files (pay attention to rm and mv, do not confuse them)
rm [target]

# copy files
cp -r [source] [destination]

# mount filesystem
mount /dev/[device name] /media/[device name]

# if want to mount a use drive (especially when automatic mounting failed)
sudo fdisk -l
mkdir /media/[device name]
sudo mount [identifiler] /media/[device name]

# unmount
unmount /media/[device name]
```


### File System Administration

```bash
# execute command as an administrator
sudo [command]

# substitutes the current user for root in the current shell
# root password can be changed by running passwd
su [user name(omit this filed will direct to root)]

# pass a command to root user, it differ from sudo as it is execute by root user
su -c '[command]'

# same as running su in the shell, but asks for the current user’s password rather than root
sudo su

# summons a shell with your $SHELL variable.
sudo -s

# virtually the same as the sudo su command with one exception: it does not directly interact with the root user.
sudo -i

# quit system administration:
exit

# check distro repositories for software updates:
sudo apt-get update

# download and install updates (update first):
sudo apt-get upgrade

# search for package in the repositories:
apt-cache search [keyword]

# get more detail on one specific package:
apt-cache show [package name]

# download and install a package:
sudo apt-get install [package name]

```


### files operation

```bash
# show all file content at once
cat [file]

# show file content, can show multiple files line by line
more [file]

# show file content, allow backward movement line by line.
less [file]

# show file content, similar to less but show line number and percentage
most [file] [file2]

# find files matching, (faster when db is updated)
sudo updatedb
locate [file]

# search through [filename] for matches to [phrase]:
grep [phrase] [filename]

## find phase in all files under current dir, (recursively, print line number, match words)
grep -rnw [phase] [dir]

# search through output of a command for [phrase]:
[command] | grep [phrase]
```


### Processes

```bash
# list all running processes in the current shell
ps

# list all running processes (-e, everythin)
ps -e

# standard system monitor showing a more extensive view of all processes and system resources
top

# like top, but with a better, cleaner interface:
htop

# Stop a process from using all system resources and lagging computer:
nice [process name]

# Kill misbehaving process (use sparingly, last resort, try 'nice' command first):
pkill [process name]

# get pid of a process by name
pidof [process name]

# kill processes, send SIGTERM (15), handled well more recommended if possible
kill [pid/processname]

# kill all process matched by name
killall [processname]

# kill processes immediately, send SIGKILL (9)
kill -9 [pid1/processname1] [pid2/processname2]

# kill process, matched by pattern
pkill [pattern]
```


### Compression and Encryption

```bash
# Make a simple compressed backup of a file or directory:
tar -cvzf [backup output.tgz] [target file or directory]

# Open a compressed .tgz or .tar.gz file:
tar -xvf [target.tgz]

# decompress split zip files, (unzip the first file)
tar -xvf [target.zip.001]

# Encrypt a file:
gpg -o [outputfilename.gpg] -c [target file]

# Decrypt a file:
gpg -o [outputfilename] -d [target.gpg]

# Zip and encrypt a directory simultaneously:
gpg-zip -o encrypted-filename.tgz.gpg -c -s file-to-be-encrypted
```