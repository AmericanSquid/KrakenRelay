"""Architecture regression checks for the explicit-service boundary."""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOTS = ("audio", "core", "dsp", "plugins", "tones")


class DependencyBoundarySmokeTests(unittest.TestCase):
    def test_services_do_not_reference_the_removed_controller(self):
        offenders = []

        for root in SERVICE_ROOTS:
            for path in (REPO_ROOT / root).rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Name) and node.id == "controller") or (
                        isinstance(node, ast.Attribute) and node.attr == "controller"
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
