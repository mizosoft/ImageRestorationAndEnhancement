# Fail the script if one command fails
set -e

pip install -r requirements.txt

if [[ ! -d 'checkpoints' ]]
then
  wget -nc https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
  unzip global_checkpoints.zip
fi

if [[ ! -d 'weights' ]]
then
  mkdir 'weights'

  while read -r line; do
    echo "Downloading $line"
    wget -nc "$line" -P weights
  done < gpen_urls.txt

  for i in $(find 'weights' -type f)
  do
      mv "$i" "$(echo "$i" | cut -d? -f1)"
  done
fi

if [[ ! -d 'models' ]]
then
#  wget -nc https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth -P models
  wget -nc https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth?dl=0 -P models

  for i in $(find 'weights' -type f)
  do
      mv "$i" "$(echo "$i" | cut -d? -f1)"
  done
fi
