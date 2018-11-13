# Starting Jupyter w/ gpu
***You must modprobe the modules on an nvidia optimus system***

modprobe nvidia
modprobe nvidia_modeset
modprobe nvidia_uvm

***Thereafter, you need to start the docker container***
sudo docker container start tf-gpu
