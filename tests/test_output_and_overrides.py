"""Output-checkbox mapping and config-override env merging.

Qt-free helpers, so unit-tested directly: the checkbox-to-process_type
ladder and layer list, and that WINMOL_CONFIG_OVERRIDES_JSON merging
respects a pre-existing value instead of clobbering it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_utils.config_overrides import (  # noqa: E402
    ENV_VAR, batch_override_env, merge,
)
from plugin_utils.output_selection import (  # noqa: E402
    NODES_LAYER, STEMS_LAYER, VECTORS_LAYER,
    gpkg_layers_for, process_type_for,
)


# -- output_selection ---------------------------------------------------

def test_process_type_ladder():
    assert process_type_for(stem=True, trees=False, nodes=False) == "Stems"
    assert process_type_for(stem=False, trees=True, nodes=False) == "Trees"
    assert process_type_for(stem=False, trees=False, nodes=True) == "Nodes"
    # Nodes wins over trees, trees wins over stem alone.
    assert process_type_for(stem=True, trees=True, nodes=True) == "Nodes"
    assert process_type_for(stem=False, trees=True, nodes=True) == "Nodes"
    assert process_type_for(stem=True, trees=True, nodes=False) == "Trees"


def test_process_type_nothing_selected_falls_back_to_stems():
    assert process_type_for(stem=False, trees=False, nodes=False) == "Stems"


def test_gpkg_layers_for_nodes_returns_all_three_in_order():
    assert gpkg_layers_for(trees=False, nodes=True) == [
        STEMS_LAYER, VECTORS_LAYER, NODES_LAYER]


def test_gpkg_layers_for_trees_returns_only_stems():
    assert gpkg_layers_for(trees=True, nodes=False) == [STEMS_LAYER]


def test_gpkg_layers_for_stem_only_is_empty():
    assert gpkg_layers_for(trees=False, nodes=False) == []


# -- config_overrides -----------------------------------------------------

def test_merge_with_no_existing_value():
    result = merge({"prediction_batch_override": 4})
    assert json.loads(result) == {"prediction_batch_override": 4}


def test_merge_respects_existing_keys_not_touched_by_the_dialog():
    existing = json.dumps({"max_cpu_workers": 3, "tile_inner_px": 4096})
    result = merge({"prediction_batch_override": 8}, existing)
    assert json.loads(result) == {
        "max_cpu_workers": 3,
        "tile_inner_px": 4096,
        "prediction_batch_override": 8,
    }


def test_merge_dialog_key_wins_over_existing():
    existing = json.dumps({"prediction_batch_override": 2})
    result = merge({"prediction_batch_override": 8}, existing)
    assert json.loads(result) == {"prediction_batch_override": 8}


def test_merge_discards_unparsable_existing_value():
    result = merge({"a": 1}, existing="not valid json {{{")
    assert json.loads(result) == {"a": 1}


def test_merge_discards_existing_non_dict_json():
    result = merge({"a": 1}, existing="[1, 2, 3]")
    assert json.loads(result) == {"a": 1}


def test_batch_override_env_auto_means_no_override():
    assert batch_override_env(0) == {}
    assert batch_override_env(None) == {}
    assert batch_override_env("not-a-number") == {}


def test_batch_override_env_pins_a_positive_batch_size():
    env = batch_override_env(8)
    assert list(env.keys()) == [ENV_VAR]
    assert json.loads(env[ENV_VAR]) == {"prediction_batch_override": 8}


def test_batch_override_env_preserves_existing_keys():
    existing = json.dumps({"max_cpu_workers": 5})
    env = batch_override_env(8, existing=existing)
    assert json.loads(env[ENV_VAR]) == {
        "max_cpu_workers": 5,
        "prediction_batch_override": 8,
    }
