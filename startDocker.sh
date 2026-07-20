xhost +
# If docker image is not build yet, simply run
#
# docker build -t winmol_analyser_docker .
#
## TODO what is this docker container for? Is this running QGIS inside docker and pass the GUI to the host machine? If so, why is it needed?
## TODO Would this work in WIndows or Mac
rm -rf winmol_venv ## TODO why would this delete the venv, it should be docker ignored
docker run --rm -it --name qgis --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix  \
    -v ./:/root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyser  \
    -e DISPLAY=unix$DISPLAY \
    winmol_analyser_docker \
    qgis
