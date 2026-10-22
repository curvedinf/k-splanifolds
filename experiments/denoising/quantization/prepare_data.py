"""Fetch and verify the pinned denoising corpus for the historical runners."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.denoising.run import dataset

if __name__=='__main__':
    dataset(Path(__file__).parent/'data')
