import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ModalRecipeTests(unittest.TestCase):
    def test_secret_dependencies_do_not_depend_on_injected_credentials(self):
        source = Path(__file__).resolve().parents[1] / 'modal_app.py'
        tree = ast.parse(source.read_text())
        node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'runtime_secrets'
                            for t in n.targets))
        # No conditional dependency graph or credential-valued Secret.from_dict.
        self.assertIsInstance(node, ast.List)
        expression = compile(ast.Expression(node), str(source), 'eval')
        secret = SimpleNamespace(from_name=lambda name, **kw: (name, kw))
        fake_modal = SimpleNamespace(Secret=secret)
        with patch.dict(os.environ, {}, clear=True):
            local = eval(expression, {'modal': fake_modal, 'os': os})
        with patch.dict(os.environ, {'HF_TOKEN': 'test-only', 'WANDB_API_KEY': 'test-only',
                                    'WANDB_ENTITY': 'test-only'}, clear=True):
            remote = eval(expression, {'modal': fake_modal, 'os': os})
        self.assertEqual(local, remote)
        self.assertEqual(len(local), 2)


if __name__ == '__main__':
    unittest.main()
