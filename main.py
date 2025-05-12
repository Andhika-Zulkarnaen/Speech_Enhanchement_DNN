# main.py

from trainers.trainer import SpeechEnhancementTrainer
import json

if __name__ == "__main__":
    with open('configs/config.json') as f:
        config = json.load(f)

    trainer = SpeechEnhancementTrainer(config)
    trainer.train()
