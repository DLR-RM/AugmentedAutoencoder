docker run -ti --rm --runtime=nvidia -v ~/share-to-docker:/shared-folder -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --privileged pytorch3d
