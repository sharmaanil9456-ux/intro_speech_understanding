# Lecture 1

Here's the code for lecture 1.  Fetch and merge this to your laptop:
```
git fetch release
git merge release/main -m "Merging release" --allow-unrelated-histories
```
Edit `homework1.py`, then check to make sure it worked by typing
```
python grade.py
```
It will print out some commentary.  The last line will be your grade.  If it's 100%, then you should upload your changes to your github repository, so that your instructor can see it:
```
git add .
git commit -m "hw1"
git push origin main
```git fetch release
git merge release/main -m "Merging release" --allow-unrelated-histories


