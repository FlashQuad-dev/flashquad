use below to compile and generate paper.pdf
docker run --rm \
  --volume $PWD/paper:/data \
  --user $(id -u):$(id -g) \
  --env JOURNAL=joss \
  openjournals/inara

