use below to compile and generate paper.pdf
docker run --rm \
  --volume $PWD/paper:/data \
  --user $(id -u):$(id -g) \
  --env JOURNAL=joss \
  openjournals/inara

To push

git add paper
git commit -m "add paper folder"
git push

git add paper
git commit -m "edit paper folder"
git push
