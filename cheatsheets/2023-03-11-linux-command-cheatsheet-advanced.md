---
layout: "post"
show-avatar: false
title: "Bash command cheat sheet (Intermediate)"
date: "2023-03-11"
tags: [linux, cheatsheet]
cover-img: /assets/img/linux.jpg
thumbnail-img: /assets/img/linux_header.jpg
---


- [awk for text display](#awk-for-text-display)
- [seed to text replace](#seed-to-text-replace)



### awk for text display

```bash
awk '{print}' text.txt

## print lines in text.txt file that matches the pattern
awk '/pattern/ {print}' text.txt

## print ith and jth column in text.txt file
awk '{print $1,$4}' text.txt

## print line number
awk '{print NR}' text.txt

## print last column
awk '{print $NF}' text.txt

## print line 2 to line 5
awk 'NR==2, NR==5 {print}' text.txt 

## change the default delimiter
awk -F ',' '{print $NF}' text.txt

## similar to the above line using split function
awk '{split($0, data, ","); print NR "-" data[5]}' text.txt
```

### seed to text replace
```bash
## replace the first occurrence small to big in text.txt
sed 's/small/big/' text.txt

## similar to above
sed 's/small/big/1' text.txt

## replace all the occurrence
sed 's/small/big/g' text.txt

## replace all the occurrence (starting form the 3rd)
sed 's/small/big/3g' text.txt
```
