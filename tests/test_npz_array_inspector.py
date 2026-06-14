from pathlib import Path

import numpy as np

from disasterbench.inspection.npz_array_inspector import inspect_npz_dataset, inspect_npz_file


def test_inspect_npz_file_reads_arrays_and_label_values(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"

    image = np.zeros((12, 32, 32), dtype=np.int16)
    aerosol = np.ones((32, 32), dtype=np.float32)
    label = np.zeros((32, 32), dtype=np.uint8)
    label[0:4, 0:4] = 1

    np.savez(npz_path, image=image, aerosol=aerosol, label=label)

    report = inspect_npz_file(npz_path)

    assert report["valid"] is True
    assert report["keys"] == ["image", "aerosol", "label"]
    assert report["arrays"]["image"]["shape"] == [12, 32, 32]
    assert report["arrays"]["aerosol"]["dtype"] == "float32"
    assert report["arrays"]["label"]["dtype"] == "uint8"
    assert report["label_value_counts"]["label"]["0"] == 1008
    assert report["label_value_counts"]["label"]["1"] == 16


def test_inspect_npz_dataset_summarizes_folder_schema(tmp_path: Path):
    dataset_root = tmp_path / "sen2fire_like"
    scene1 = dataset_root / "scene1"
    scene2 = dataset_root / "scene2"
    scene1.mkdir(parents=True)
    scene2.mkdir(parents=True)

    image = np.zeros((12, 32, 32), dtype=np.int16)
    aerosol = np.ones((32, 32), dtype=np.float32)

    label_positive = np.zeros((32, 32), dtype=np.uint8)
    label_positive[0:2, 0:2] = 1

    label_zero = np.zeros((32, 32), dtype=np.uint8)

    np.savez(scene1 / "patch_001.npz", image=image, aerosol=aerosol, label=label_positive)
    np.savez(scene2 / "patch_002.npz", image=image, aerosol=aerosol, label=label_zero)

    output_path = tmp_path / "report.json"

    report = inspect_npz_dataset(
        dataset_root=dataset_root,
        output_path=output_path,
    )

    assert report["total_npz_files"] == 2
    assert report["inspected_npz_files"] == 2
    assert report["folder_counts"]["scene1"] == 1
    assert report["folder_counts"]["scene2"] == 1
    assert report["key_combinations"]["('aerosol', 'image', 'label')"] == 2
    assert report["array_shapes_by_key"]["image"]["(12, 32, 32)"] == 2
    assert report["array_shapes_by_key"]["label"]["(32, 32)"] == 2
    assert report["array_dtypes_by_key"]["label"]["uint8"] == 2
    assert report["label_value_counts_by_key"]["label"]["0"] == 2044
    assert report["label_value_counts_by_key"]["label"]["1"] == 4
    assert report["files_with_positive_label_by_key"]["label"] == 1
    assert report["files_with_only_zero_label_by_key"]["label"] == 1
    assert report["read_error_count"] == 0
    assert output_path.exists()
