from typing import List, Optional

from PyQt5.QtCore import QVariant
from qgis.core import QgsFeature, QgsField, QgsFields, QgsGeometry, QgsPointXY
from shapely import set_srid, to_wkb

from classes.Stem import Stem


class FeatureFactory:
    _stem_fields = QgsFields()
    _node_vector_fields = QgsFields()

    def __init__(self):
        self._create_stem_fields()
        self._create_node_vector_fields()

    def _create_stem_fields(self):
        self._stem_fields.append(QgsField('id', QVariant.Int))
        self._stem_fields.append(QgsField('diameters', QVariant.List))
        self._stem_fields.append(QgsField('volumes', QVariant.List))
        self._stem_fields.append(QgsField('vector', QVariant.List))
        self._stem_fields.append(QgsField('length', QVariant.Double))
        self._stem_fields.append(QgsField('volume', QVariant.Double))
        self._stem_fields.append(QgsField('start', QVariant.Point))
        self._stem_fields.append(QgsField('stop', QVariant.Point))
        self._stem_fields.append(QgsField('crs', QVariant.String))

    def _create_node_vector_fields(self):
        self._node_vector_fields.append(QgsField('stem_id', QVariant.Int))
        self._node_vector_fields.append(QgsField('node_id', QVariant.Int))
        self._node_vector_fields.append(QgsField('vector', QVariant.List))
        self._node_vector_fields.append(QgsField('diameter', QVariant.Double))

    def _extract_crs_code(self, crs: Optional[str]) -> int:
        # Extract the numerical EPSG code from the CRS string
        if crs and crs.startswith("EPSG:"):
            return int(crs.split(":")[1])
        return None  # Return undefined if no EPSG code is available

    # We define that the main geometry of a stem is its linestring

    def create_stem_feature(self, stem: Stem) -> QgsFeature:
        crs_code = self._extract_crs_code(stem.crs)  # Get EPSG code dynamically
        feat = QgsFeature(self._stem_fields)
        geom = QgsGeometry()
        geom.fromWkb(to_wkb(set_srid(stem.path, crs_code)))
        feat.setGeometry(geom)

        # set attribute values
        feat.setAttribute('id', stem.stem_id)
        feat.setAttribute('diameters', stem.segment_diameter_list)
        feat.setAttribute('volumes', stem.segment_volume_list)
        feat.setAttribute('vector', stem.vector)
        feat.setAttribute('length', stem.length())
        feat.setAttribute('volume', stem.volume())
        feat.setAttribute('crs', stem.crs if stem.crs else "Undefined")
        feat.setAttribute(
            'start',
            QgsGeometry.fromPointXY(QgsPointXY(stem.start.x, stem.start.y))
        )
        feat.setAttribute(
            'stop',
            QgsGeometry.fromPointXY(QgsPointXY(stem.stop.x, stem.stop.y))
        )

        return feat

    def create_subsidiary_features(
            self,
            stem: Stem
    ) -> List[QgsFeature]:
        if len(stem.get_nodes()) == 0:
            return []

        crs_code = self._extract_crs_code(stem.crs)  # Get EPSG code dynamically
        node_features = []
        for node in stem.get_nodes():
            feat = QgsFeature(self._node_vector_fields)
            geom = QgsGeometry()
            geom.fromWkb(to_wkb(set_srid(node.geom, crs_code)))
            feat.setGeometry(geom)

            # set attribute values
            feat.setAttribute('stem_id', node.stem_id)
            feat.setAttribute('node_id', node.node_id)
            feat.setAttribute('vector', node.vector)
            feat.setAttribute('diameter', node.diameter)
            node_features.append(feat)

        return node_features
