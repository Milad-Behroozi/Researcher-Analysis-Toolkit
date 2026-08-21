from __future__ import annotations

import ast
import csv
import json
import math
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module_targets() -> list[str]:
    tree = ast.parse((ROOT / "launcher.py").read_text(encoding="utf-8-sig"))
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODULES":
                    value = ast.literal_eval(node.value)
    if value is None:
        raise AssertionError("MODULES was not found in launcher.py")

    targets: list[str] = []

    def collect(item) -> None:
        if isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, str) and item.endswith(".py"):
            targets.append(item)

    collect(value)
    return targets


class RepositoryValidationTests(unittest.TestCase):
    def test_all_root_python_files_parse(self) -> None:
        files = sorted(ROOT.glob("*.py"))
        self.assertGreaterEqual(len(files), 15)
        for path in files:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_every_launcher_target_exists(self) -> None:
        targets = module_targets()
        self.assertGreaterEqual(len(targets), 15)
        for target in targets:
            with self.subTest(target=target):
                self.assertTrue((ROOT / target).is_file())

    def test_required_repository_documents_exist(self) -> None:
        required = [
            "README.md",
            "USER_GUIDE.md",
            "CITATION.cff",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "requirements.txt",
            "run.bat",
            "examples/validation_manifest.json",
            "packaging/build_windows.bat",
            "packaging/download_python_runtime.ps1",
        ]
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_validation_manifest_inputs_exist(self) -> None:
        manifest_path = ROOT / "examples" / "validation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest["examples"]), 6)
        for example in manifest["examples"]:
            with self.subTest(example=example["id"]):
                self.assertTrue((manifest_path.parent / example["input"]).is_file())

    def test_tafel_csv_is_finite_and_complete(self) -> None:
        path = ROOT / "examples" / "tafel" / "LSV_oxygen_deficient_tungsten_oxide_example.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["Potential_V", "Current_A"])
            rows = list(reader)
        self.assertEqual(len(rows), 1800)
        for row in rows:
            self.assertTrue(math.isfinite(float(row["Potential_V"])))
            self.assertTrue(math.isfinite(float(row["Current_A"])))

    def test_synthetic_eis_files(self) -> None:
        eis_dir = ROOT / "examples" / "eis"
        metadata = json.loads((eis_dir / "benchmark_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["generating_circuit"], "R0-p(R1,CPE1)-p(R2,CPE2)")
        self.assertEqual(metadata["random_seed"], 20260723)
        for filename in metadata["datasets"]:
            with self.subTest(filename=filename):
                with (eis_dir / filename).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, ["frequency_Hz", "Z_real_ohm", "Z_imag_ohm"])
                    rows = list(reader)
                self.assertEqual(len(rows), metadata["point_count"])
                frequencies = [float(row["frequency_Hz"]) for row in rows]
                self.assertTrue(all(a > b for a, b in zip(frequencies, frequencies[1:])))
                for row in rows:
                    self.assertTrue(all(math.isfinite(float(value)) for value in row.values()))

    def test_example_workbooks_are_valid_xlsx_containers(self) -> None:
        workbooks = [
            ROOT / "examples" / "trpl" / "TRPL_example.xlsx",
            ROOT / "examples" / "tauc" / "Tauc_diffuse_reflectance_example.xlsx",
        ]
        for path in workbooks:
            with self.subTest(path=path.name):
                self.assertTrue(zipfile.is_zipfile(path))
                with zipfile.ZipFile(path) as archive:
                    names = set(archive.namelist())
                self.assertIn("[Content_Types].xml", names)
                self.assertIn("xl/workbook.xml", names)


if __name__ == "__main__":
    unittest.main()

