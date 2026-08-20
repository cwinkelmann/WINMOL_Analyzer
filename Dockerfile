FROM qgis/qgis:final-3_28_13

RUN apt update && apt install -y python3.10-venv

RUN ln -s /usr/bin/python3 /usr/bin/python
