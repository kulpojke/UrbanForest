Setup git:

```
git remote add origin https://github.com/kulpojke/UrbanForest
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=43200'
git config --global user.email michaelhuggins@protonmail.com
git config --global user.name michael
rm .config/configstore/update-notifier-npm.json
rm .local/share/jupyter/runtime/nbserver-1.json
git pull origin
git checkout main

conda update numpy
```
