xhost +
# If docker image is not build yet, simply run
#
# docker build -t winmol_analyser_docker .
#
rm -rf winmol_venv
docker run --rm -it --name qgis --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix  \
    -v ./:/root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyser  \
    -e DISPLAY=unix$DISPLAY \
    winmol_analyser_docker \
    qgis
