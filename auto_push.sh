#!/bin/bash
cd /workspace/Janus_fine_tuning || exit
timestamp=$(date "+%Y-%m-%d %H:%M:%S")

git add .
git commit -m "$timestamp"
git push origin master
