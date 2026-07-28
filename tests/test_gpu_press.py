import unittest
from unittest.mock import Mock, patch

import gpu_press


class GpuPressTest(unittest.TestCase):
    def setUp(self):
        gpu_press.burn_procs.clear()
        gpu_press.gpu_inventory.clear()
        gpu_press.active_gpu_indices = []
        gpu_press.running = False

    def test_query_gpus_parses_every_nvidia_smi_row(self):
        output = (
            "0, GPU-first, NVIDIA GeForce RTX 3070, 595.71.05, "
            "50, 120, 220, 90, 1024, 8192, 1350\n"
            "1, GPU-second, NVIDIA GeForce RTX 3070, 595.71.05, "
            "52, 125, 220, 91, 2048, 8192, 1365\n"
        ).encode()

        with patch("gpu_press.subprocess.check_output", return_value=output):
            samples = gpu_press.query_gpus()

        self.assertEqual([sample["gpu_index"] for sample in samples], [0, 1])
        self.assertEqual(samples[0]["sm_clock"], 1350.0)
        self.assertEqual(samples[1]["gpu_uuid"], "GPU-second")

    def test_start_burn_binds_one_process_to_each_selected_uuid(self):
        gpu_press.gpu_inventory.update({
            0: {"uuid": "GPU-first"},
            2: {"uuid": "GPU-third"},
        })
        processes = [Mock(), Mock()]

        with patch("gpu_press.is_binary_ready", return_value=True), patch(
                "gpu_press.subprocess.Popen", side_effect=processes) as popen:
            ok, message = gpu_press.start_burn(60, 2, [0, 2])

        self.assertTrue(ok)
        self.assertEqual(message, "已启动 2 张 GPU")
        self.assertEqual(
            [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list],
            ["GPU-first", "GPU-third"],
        )

    def test_start_api_requires_unique_integer_gpu_indices(self):
        client = gpu_press.app.test_client()

        empty = client.post("/api/start", json={"gpu_indices": []})
        duplicate = client.post("/api/start", json={"gpu_indices": [0, 0]})
        boolean = client.post("/api/start", json={"gpu_indices": [True]})

        self.assertEqual(empty.status_code, 400)
        self.assertEqual(duplicate.status_code, 400)
        self.assertEqual(boolean.status_code, 400)


if __name__ == "__main__":
    unittest.main()
