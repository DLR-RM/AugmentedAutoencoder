docker run -ti --rm --gpus all -v /home/hampus/vision:/shared-folder -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --privileged pytorch3d
