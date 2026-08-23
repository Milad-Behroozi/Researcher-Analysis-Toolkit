from __future__ import annotations

import ast
import csv
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module_targets() -> list[str]:
    tree = ast.parse(
        (ROOT / "launcher.py").read_text(encoding="utf-8-sig")
    )
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
                ast.parse(
                    path.read_text(encoding="utf-8-sig"),
                    filename=str(path),
                )

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
        manifest_path = (
            ROOT / "examples" / "validation_manifest.json"
        )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["manifest_version"], "2.0")
        self.assertGreaterEqual(len(manifest["examples"]), 7)

        example_ids = [
            example["id"]
            for example in manifest["examples"]
        ]
        self.assertEqual(
            len(example_ids),
            len(set(example_ids)),
        )

        for example in manifest["examples"]:
            with self.subTest(example=example["id"]):
                input_path = (
                    manifest_path.parent / example["input"]
                )
                self.assertTrue(input_path.is_file())

    def test_tafel_csv_is_finite_and_complete(self) -> None:
        path = (
            ROOT
            / "examples"
            / "tafel"
            / "LSV_oxygen_deficient_tungsten_oxide_example.csv"
        )

        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(
                reader.fieldnames,
                ["Potential_V", "Current_A"],
            )
            rows = list(reader)

        self.assertEqual(len(rows), 1800)

        for row in rows:
            self.assertTrue(
                math.isfinite(float(row["Potential_V"]))
            )
            self.assertTrue(
                math.isfinite(float(row["Current_A"]))
            )

    def test_synthetic_eis_files(self) -> None:
        eis_dir = ROOT / "examples" / "eis"
        metadata_path = (
            eis_dir / "benchmark_metadata.json"
        )
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        self.assertEqual(metadata["schema_version"], "2.0")
        self.assertEqual(metadata["random_seed"], 20260723)
        self.assertEqual(metadata["point_count"], 50)
        self.assertEqual(
            metadata["frequency_range_Hz"],
            [0.01, 1000000.0],
        )
        self.assertEqual(
            metadata["frequency_distribution"],
            "logarithmic, descending",
        )

        expected_files = {
            "EIS_single_relaxation_noise0.csv",
            "EIS_single_relaxation_noise1.csv",
            "EIS_single_relaxation_noise3.csv",
            "EIS_single_relaxation_noise5.csv",
            "EIS_two_relaxation_noise0.csv",
            "EIS_two_relaxation_noise1.csv",
            "EIS_two_relaxation_noise3.csv",
            "EIS_two_relaxation_noise5.csv",
            "EIS_diffusion_noise0.csv",
            "EIS_diffusion_noise1.csv",
            "EIS_diffusion_noise3.csv",
            "EIS_diffusion_noise5.csv",
        }

        expected_circuits = {
            "single_relaxation":
                "R0-p(R1,CPE1)",
            "two_relaxation":
                "R0-p(R1,CPE1)-p(R2,CPE2)",
            "diffusion":
                "R0-p(R1,CPE1)-W1",
        }

        circuits = metadata["circuits"]
        self.assertEqual(len(circuits), 3)

        found_files: set[str] = set()
        found_circuits: dict[str, str] = {}

        for circuit in circuits:
            circuit_name = circuit["name"]

            self.assertIn(
                circuit_name,
                expected_circuits,
            )
            self.assertEqual(
                circuit["generating_circuit"],
                expected_circuits[circuit_name],
            )

            found_circuits[circuit_name] = (
                circuit["generating_circuit"]
            )

            datasets = circuit["datasets"]
            self.assertEqual(len(datasets), 4)

            noise_fractions = []

            for filename, dataset_info in datasets.items():
                with self.subTest(
                    circuit=circuit_name,
                    filename=filename,
                ):
                    self.assertNotIn(
                        filename,
                        found_files,
                    )
                    found_files.add(filename)

                    noise_fractions.append(
                        float(
                            dataset_info["noise_fraction"]
                        )
                    )

                    path = eis_dir / filename
                    self.assertTrue(path.is_file())

                    with path.open(
                        newline="",
                        encoding="utf-8-sig",
                    ) as handle:
                        reader = csv.DictReader(handle)
                        self.assertEqual(
                            reader.fieldnames,
                            [
                                "frequency_Hz",
                                "Z_real_ohm",
                                "Z_imag_ohm",
                            ],
                        )
                        rows = list(reader)

                    self.assertEqual(
                        len(rows),
                        metadata["point_count"],
                    )

                    frequencies = [
                        float(row["frequency_Hz"])
                        for row in rows
                    ]

                    self.assertTrue(
                        math.isclose(
                            frequencies[0],
                            1000000.0,
                            rel_tol=1e-9,
                        )
                    )
                    self.assertTrue(
                        math.isclose(
                            frequencies[-1],
                            0.01,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                    )
                    self.assertTrue(
                        all(
                            first > second
                            for first, second in zip(
                                frequencies,
                                frequencies[1:],
                            )
                        )
                    )

                    log_frequencies = [
                        math.log10(frequency)
                        for frequency in frequencies
                    ]
                    log_steps = [
                        second - first
                        for first, second in zip(
                            log_frequencies,
                            log_frequencies[1:],
                        )
                    ]

                    self.assertTrue(
                        all(
                            math.isclose(
                                step,
                                log_steps[0],
                                rel_tol=1e-8,
                                abs_tol=1e-8,
                            )
                            for step in log_steps
                        )
                    )

                    for row in rows:
                        values = [
                            float(
                                row["frequency_Hz"]
                            ),
                            float(
                                row["Z_real_ohm"]
                            ),
                            float(
                                row["Z_imag_ohm"]
                            ),
                        ]
                        self.assertTrue(
                            all(
                                math.isfinite(value)
                                for value in values
                            )
                        )

            self.assertEqual(
                sorted(noise_fractions),
                [0.0, 0.01, 0.03, 0.05],
            )

        self.assertEqual(
            found_files,
            expected_files,
        )
        self.assertEqual(
            found_circuits,
            expected_circuits,
        )

        ranking_files = set(
            metadata[
                "automatic_topology_ranking_datasets"
            ]
        )
        self.assertEqual(
            ranking_files,
            {
                "EIS_single_relaxation_noise0.csv",
                "EIS_two_relaxation_noise3.csv",
                "EIS_diffusion_noise5.csv",
            },
        )
        self.assertTrue(
            ranking_files.issubset(found_files)
        )

    def test_example_csv_files_are_finite(self) -> None:
        csv_files = [
            (
                ROOT
                / "examples"
                / "trpl"
                / "TRPL_example.csv",
                [
                    "Time(ns)",
                    "PL intensity(a.u.)",
                ],
            ),
            (
                ROOT
                / "examples"
                / "tauc"
                / "Tauc_diffuse_reflectance_example.csv",
                [
                    "Wavelength nm.",
                    "R",
                ],
            ),
        ]

        for path, expected_columns in csv_files:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

                with path.open(
                    newline="",
                    encoding="utf-8-sig",
                ) as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(
                        reader.fieldnames,
                        expected_columns,
                    )
                    rows = list(reader)

                self.assertGreater(len(rows), 10)

                for row in rows:
                    values = [
                        float(row[column])
                        for column in expected_columns
                    ]
                    self.assertTrue(
                        all(
                            math.isfinite(value)
                            for value in values
                        )
                    )


if __name__ == "__main__":
    unittest.main()
