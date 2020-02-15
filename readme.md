# CS5242 Project 2019/2020 Semester 2

## Dataset

The Kaggle competition is [here](https://www.kaggle.com/c/cs5242project/overview). The original Breakfasts dataset is 
found [here](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).


## Extras

### Installing Video Codecs

To install video codecs on Ubuntu 18.04, run the following commands:
```bash
sudo apt update
sudo apt install libdvdnav4 libdvdread4 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libdvd-pkg
sudo apt install ubuntu-restricted-extras
```
If `libdvd-pkg` is broken, run this:
```shell script
sudo dpkg-reconfigure libdvd-pkg
```


