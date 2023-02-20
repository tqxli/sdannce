import unittest
from unittest.mock import patch
import dannce.cli as cli
import os

SDANNCE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SDANNCE_PROJECT_FOLDER = os.path.join(SDANNCE_FOLDER, "tests", "2021_07_06_M3_M6")
TEST_SDANNCE_PREDICT_PROJECT_FOLDER = os.path.join(
    SDANNCE_FOLDER, "tests", "2021_07_05_M4_M7"
)
TEST_COM_TRAIN_PROJECT_FOLDER = os.path.join(
    SDANNCE_FOLDER, "tests", "2021_07_06_M3_M6"
)
TEST_COM_CONFIG = os.path.join(SDANNCE_FOLDER, "configs", "com_mouse_config.yaml")
TEST_SDANNCE_CONFIG = os.path.join(SDANNCE_FOLDER, "configs", "sdannce_rat_config.yaml")


class TestComTrain(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_COM_TRAIN_PROJECT_FOLDER)

    def test_com_train(self):
        args = ["com-train", TEST_COM_CONFIG, "--epochs=2"]
        with patch("sys.argv", args):
            cli.com_train_cli()

    def test_com_train_mono(self):
        args = ["com-train", TEST_COM_CONFIG, "--mono=True", "--epochs=2"]
        with patch("sys.argv", args):
            cli.com_train_cli()


class TestComPredict(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_COM_TRAIN_PROJECT_FOLDER)

    def test_com_predict(self):
        args = [
            "com-predict",
            TEST_COM_CONFIG,
            "--com-predict-weights=./COM/train_test/checkpoint-epoch2.pth",
            "--com-predict-dir=./COM/predict_test",
            "--max-num-samples=10",
            "--batch-size=1",
        ]
        with patch("sys.argv", args):
            cli.com_predict_cli()


class TestSdannceTrain(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_SDANNCE_PROJECT_FOLDER)

    def test_dannce_train(self):
        args = [
            "sdannce-train",
            TEST_SDANNCE_CONFIG,
            "--epochs=2",
            "--train-mode=finetune",
            "--dannce-finetune-weights=../weights/DANNCE_comp_pretrained_single+r7m.pth",
            "--net-type=compressed_dannce",
            "--use-npy=True",
        ]
        with patch("sys.argv", args):
            cli.sdannce_train_cli()


class TestSdanncePredict(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_SDANNCE_PREDICT_PROJECT_FOLDER)

    def test_dannce_predict(self):
        args = [
            "sdannce-predict",
            TEST_SDANNCE_CONFIG,
            "--dannce-predict-model=../weights/SDANNCE_gcn_bsl_FM_ep100.pth",
            "--dannce-predict-dir=./DANNCE/predict_test",
            "--com-file=./COM/predict01/com3d.mat",
            "--max-num-samples=10",
            "--batch-size=1",
        ]
        with patch("sys.argv", args):
            cli.sdannce_predict_cli()


if __name__ == "__main__":
    unittest.main()
