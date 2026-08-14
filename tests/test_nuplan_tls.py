from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nuplan_tls  # noqa: E402


class NuPlanTlsTest(unittest.TestCase):
    def test_uses_system_context_when_it_has_trusted_roots(self) -> None:
        system_context = mock.Mock()
        system_context.get_ca_certs.return_value = [{"subject": "fixture"}]
        with mock.patch("nuplan_tls.ssl.create_default_context", return_value=system_context) as create:
            self.assertIs(nuplan_tls.verified_https_context(), system_context)
        create.assert_called_once_with()

    def test_falls_back_to_certifi_without_disabling_verification(self) -> None:
        empty_context = mock.Mock()
        empty_context.get_ca_certs.return_value = []
        certifi_context = mock.Mock()
        certifi_context.get_ca_certs.return_value = [{"subject": "fixture"}]
        with mock.patch(
            "nuplan_tls.ssl.create_default_context",
            side_effect=[empty_context, certifi_context],
        ) as create:
            with mock.patch("certifi.where", return_value="/fixture/cacert.pem"):
                with mock.patch("nuplan_tls.Path.is_file", return_value=True):
                    self.assertIs(nuplan_tls.verified_https_context(), certifi_context)
        self.assertEqual(create.call_args_list, [mock.call(), mock.call(cafile="/fixture/cacert.pem")])


if __name__ == "__main__":
    unittest.main()
