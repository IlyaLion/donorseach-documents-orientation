import argparse
import warnings

from config.config import get_config
from training.training import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        help='Path to config file')
    args = parser.parse_args()
    config_filename = args.config_filename
    config = get_config(config_filename)
    
    train_model(config)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()