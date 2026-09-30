#!/bin/bash
# NOTE: This script should run at /root/ directory (not /)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "NOTE: This script should run at /root/ directory (not /)"

VERSION="V1.0"
SPLITNO="all"
while getopts ":hs:" opt; do
  case $opt in
    h) echo -e "CD-FSVOD Setup $VERSION"
       echo -e "Usage: $0 -s splitN"
       echo -e "    $0 -h \t\t\t: show this help menu"
       echo -e "    $0 -s {s1, s2, s3, s4, vot, all} \t: specify model to download (default 'all')"
       exit
       ;;
    s) echo "Split specified as $OPTARG"
       SPLITNO=$OPTARG
       ;;
  esac
done

# Install necessary packages
echo "Installing necessary packages"
apt install ripgrep -y &> /dev/null
apt install unzip -y &> /dev/null
pip install gdown

# Download pretrained BHRL base models
if [[ $SPLITNO == "all" ]]; then
  echo "Downloading all BHRL pretrained models"
  gdown https://drive.google.com/uc?id=1tl7O7m7SAaBZIi0u-z4uFUNFwUtoVm3o # Split 1 model
  gdown https://drive.google.com/uc?id=1D-PUQPH5NELTf52xUpGXO5jfxpa76pc8 # Split 2 model
  gdown https://drive.google.com/uc?id=1GRyXANf60WaJp7UMqTEAGnUxftm_QLrG # Split 3 model
  gdown https://drive.google.com/uc?id=1sVSDI0aXNIPTgOPFDhpndcTwRR6mSlSU # Split 4 model
  gdown https://drive.google.com/uc?id=1TdmOfNoAYe9HTXozeBsG9inwyUS2J-az # VOC     model
elif [[ $SPLITNO == "s1" ]];then
  echo "Downloading Split 1 BHRL pretrained model"
  gdown https://drive.google.com/uc?id=1tl7O7m7SAaBZIi0u-z4uFUNFwUtoVm3o # Split 1 model
elif [[ $SPLITNO == "s2" ]];then
  echo "Downloading Split 2 BHRL pretrained model"
  gdown https://drive.google.com/uc?id=1D-PUQPH5NELTf52xUpGXO5jfxpa76pc8 # Split 2 model
elif [[ $SPLITNO == "s3" ]];then
  echo "Downloading Split 3 BHRL pretrained model"
  gdown https://drive.google.com/uc?id=1GRyXANf60WaJp7UMqTEAGnUxftm_QLrG # Split 3 model
elif [[ $SPLITNO == "s4" ]];then
  echo "Downloading Split 4 BHRL pretrained model"
  gdown https://drive.google.com/uc?id=1sVSDI0aXNIPTgOPFDhpndcTwRR6mSlSU # Split 4 model
elif [[ $SPLITNO == "vot" ]];then
  echo "Downloading VOC BHRL pretrained model"
  gdown https://drive.google.com/uc?id=1TdmOfNoAYe9HTXozeBsG9inwyUS2J-az # VOC     model
fi

# FSVOD-500 within-domain base model trained in the thesis
echo "Downloading FSVOD-500 base model"
wget -q --show-progress https://github.com/msprITU/CD-FSVOD/releases/download/v1.0/model_fsvod.pth

# Download code and annotation archives
echo "Downloading BHRL.zip"
gdown https://drive.google.com/uc?id=18XrJsSBGeVt5VDRaL2t7kFp_HqDMDjMR # BHRL.zip
echo "Downloading artoxor_annotation.zip"
gdown https://drive.google.com/uc?id=17Y-todgImNTwR-ixqJTgbCw5k9r_XI11 # artoxor_annotation.zip
echo "Downloading dior_annotation.zip"
gdown https://drive.google.com/uc?id=1d45B4APTjyJV7Ha5YhrBIM1JK1K5bTUR # dior_annotation.zip
echo "Downloading uodd_annotation.zip"
gdown https://drive.google.com/uc?id=1gS5xAm8EjDsjnO0a73Gb4xVYL_xapSit # uodd_annotation.zip
echo "Copying annotation archives shipped with the repository"
cp "$SCRIPT_DIR/../data/vot_annotation.zip" .
cp "$SCRIPT_DIR/../data/fsvod500_annotation_cb.zip" .

# Unzip them
echo "Unzipping files"
unzip BHRL.zip
unzip artoxor_annotation.zip
unzip dior_annotation.zip
unzip uodd_annotation.zip
unzip vot_annotation.zip
unzip fsvod500_annotation_cb.zip

# Move to necessary directories
echo "Moving files"
mv artoxor_annotation BHRL/
mv dior_annotation BHRL/
mv uodd_annotation BHRL/
mv vot_annotation BHRL/
mv fsvod500_annotation_cb BHRL/

# Fix hardcoded paths in annotation files and helper scripts
echo "Rewriting directory paths"
rg "/content/drive/MyDrive/BHRL" -l | xargs -i sed -i 's#/content/drive/MyDrive/BHRL#/root/BHRL#g' {}
rg "/truba/home/ionur/BHRL" -l | xargs -i sed -i 's#/truba/home/ionur/BHRL#/root/BHRL#g' {}
rg "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT" -l | xargs -i sed -i 's#/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT#/root/BHRL/VOTIMAGES#g' {}

# Apply repository patches over the BHRL helper scripts
cp "$SCRIPT_DIR/../patches/find_update_frame.py" BHRL/scripts/
cp "$SCRIPT_DIR/../patches/find_update_frame_absent.py" BHRL/scripts/
cp "$SCRIPT_DIR/../patches/split_seq_imgs.py" BHRL/scripts/
cp "$SCRIPT_DIR/../data/fsvod_train.json" BHRL/

# Move models into checkpoints directory
mkdir -p BHRL/checkpoints
mv *.pth BHRL/checkpoints/

# Success message
echo "Configuration successful"
echo "Continue with setup/1_install_dependencies.ipy"
