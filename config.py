# config.py

config = {
    "data_config": {
        "base_dir": "data",
        "noisy_dir": "data/noisy",
        "clean_dir": "data/clean",
        "sample_rate": 16000
    },
    "training_config": {
        "batch_size": 8,
        "epochs": 35,
        "learning_rate": 0.001,
        "checkpoint_dir": "saved_models",
        "log_dir": "logs"
    },
    "feature_config": {
        "n_fft": 512,
        "hop_length": 256
    }
}
