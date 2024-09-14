import unittest
from unittest.mock import patch
import dannce.cli as cli
from dannce.engine.utils.vis import visualize_pose_predictions
import os

SDANNCE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SDANNCE_PROJECT_FOLDER = os.path.join(SDANNCE_FOLDER, "tests", "2021_07_06_M3_M6")
TEST_SDANNCE_PREDICT_PROJECT_FOLDER = os.path.join(
    SDANNCE_FOLDER, "tests", "2021_07_05_M4_M7"
)
TEST_DANNCE_PREDICT_PROJECT_FOLDER = os.path.join(
    SDANNCE_FOLDER, "tests", "2021_07_06_M3_M6"
)
TEST_COM_TRAIN_PROJECT_FOLDER = os.path.join(
    SDANNCE_FOLDER, "tests", "2021_07_06_M3_M6"
)
TEST_COM_CONFIG = os.path.join(SDANNCE_FOLDER, "configs", "com_mouse_config.yaml")
TEST_DANNCE_CONFIG = os.path.join(SDANNCE_FOLDER, "configs", "dannce_rat_config.yaml")
TEST_SDANNCE_CONFIG = os.path.join(SDANNCE_FOLDER, "configs", "sdannce_rat_config.yaml")


def test_main(args: list):
    with patch("sys.argv", args):
        cli.main()


class TestComTrain(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_COM_TRAIN_PROJECT_FOLDER)

    def test_com_train(self):
        args = ["dannce", "train", "com", TEST_COM_CONFIG, "--epochs=2"]
        test_main(args)

    def test_com_train_mono(self):
        args = ["dannce", "train", "com", TEST_COM_CONFIG, "--mono=True", "--epochs=2", "--com-train-dir=./COM/train_mono_test"]
        test_main(args)


class TestComPredict(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_COM_TRAIN_PROJECT_FOLDER)

    def test_com_predict(self):
        args = [
            "dannce",
            "predict",
            "com",
            TEST_COM_CONFIG,
            "--com-predict-weights=./COM/train_test/checkpoint-epoch2.pth",
            "--com-predict-dir=./COM/predict_test",
            "--max-num-samples=10",
            "--batch-size=1",
        ]
        test_main(args)

    def test_com_predict_mono(self):
        args = [
            "dannce",
            "predict",
            "com",
            TEST_COM_CONFIG,
            "--com-predict-weights=./COM/train_mono_test/checkpoint-epoch2.pth",
            "--mono=True",
            "--com-predict-dir=./COM/predict_mono_test",
            "--max-num-samples=10",
            "--batch-size=1",
        ]
        test_main(args)


class TestSdannceTrain(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_SDANNCE_PROJECT_FOLDER)

    def test_sdannce_train(self):
        args = [
            "dannce",
            "train",
            "sdannce",
            TEST_SDANNCE_CONFIG,
            "--epochs=2",
            "--train-mode=finetune",
            "--dannce-finetune-weights=../weights/DANNCE_comp_pretrained_single+r7m.pth",
            "--net-type=compressed_dannce",
            "--use-npy=True",
        ]
        test_main(args)


class TestSdanncePredict(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_SDANNCE_PREDICT_PROJECT_FOLDER)

    def test_dannce_predict(self):
        args = [
            "dannce",
            "predict",
            "sdannce",
            TEST_SDANNCE_CONFIG,
            "--dannce-predict-model=../weights/SDANNCE_gcn_bsl_FM_ep100.pth",
            "--dannce-predict-dir=./SDANNCE/predict_test",
            "--com-file=./COM/predict01/com3d.mat",
            "--max-num-samples=10",
            "--batch-size=1",
        ]
        test_main(args)
        
        visualize_pose_predictions(
            exproot=TEST_SDANNCE_PREDICT_PROJECT_FOLDER,
            expfolder='SDANNCE/predict_test',
            datafile='save_data_AVG0.mat',
            n_frames=10,
            start_frame=0,
            cameras="1",
            animal="rat23",
            n_animals=2,
        )


class TestDannceTrain(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_SDANNCE_PROJECT_FOLDER)

    def test_dannce_train(self):
        args = [
            "dannce",
            "train",
            "dannce",
            TEST_DANNCE_CONFIG,
            "--epochs=2",
            "--train-mode=finetune",
            "--dannce-finetune-weights=../weights/DANNCE_comp_pretrained_single+r7m.pth",
            "--net-type=compressed_dannce",
            "--use-npy=True",
        ]
        test_main(args)


class TestDanncePredict(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_DANNCE_PREDICT_PROJECT_FOLDER)

    def test_dannce_predict(self):
        args = [
            "dannce",
            "predict",
            "dannce",
            TEST_DANNCE_CONFIG,
            "--dannce-predict-model=../weights/DANNCE_comp_pretrained_single+r7m.pth",
            "--dannce-predict-dir=./DANNCE/predict_test",
            "--com-file=./COM/predict01/instance0com3d.mat",
            "--max-num-samples=10",
            "--batch-size=1",
        ]
        test_main(args)
        visualize_pose_predictions(
            exproot=TEST_DANNCE_PREDICT_PROJECT_FOLDER,
            expfolder='DANNCE/predict_test',
            datafile='save_data_AVG0.mat',
            n_frames=10,
            start_frame=0,
            cameras="1",
            animal="rat23",
            n_animals=1,
        )


class TestDanncePredictMulti(unittest.TestCase):
    def setUp(self):
        os.chdir(TEST_SDANNCE_PREDICT_PROJECT_FOLDER)

    def test_dannce_predict(self):
        args = [
            "dannce",
            "predict",
            "dannce",
            TEST_DANNCE_CONFIG,
            "--dannce-predict-model=../weights/DANNCE_comp_pretrained_single+r7m.pth",
            "--dannce-predict-dir=./DANNCE/predict_test_multi",
            "--com-file=./COM/predict01/com3d.mat",
            "--max-num-samples=10",
            "--batch-size=1",
            "--n-instances=2",
        ]
        test_main(args)

        visualize_pose_predictions(
            exproot=TEST_SDANNCE_PREDICT_PROJECT_FOLDER,
            expfolder='DANNCE/predict_test_multi',
            datafile='save_data_AVG0.mat',
            n_frames=10,
            start_frame=0,
            cameras="1",
            animal="rat23",
            n_animals=2,
        )


if __name__ == "__main__":
    import shutil

    # clean up any previous test outputs
    def clean_DANNCE_outputs(folder):
        dannce_folder = f"{folder}/DANNCE"
        sdannce_folder = f"{folder}/SDANNCE"
        
        if os.path.exists(dannce_folder):
            shutil.rmtree(dannce_folder)
        if os.path.exists(sdannce_folder):
            shutil.rmtree(sdannce_folder)
    
    clean_DANNCE_outputs(TEST_SDANNCE_PREDICT_PROJECT_FOLDER)
    clean_DANNCE_outputs(TEST_DANNCE_PREDICT_PROJECT_FOLDER)
    
    unittest.main()
