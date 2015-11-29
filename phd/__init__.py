import warnings
# Ignore a few annoying warnings
warnings.filterwarnings("ignore", message="axes.color_cycle is deprecated.*")
warnings.filterwarnings("ignore", message="Turning off units")
warnings.filterwarnings("ignore", message=".*sparse matrix patch.*")

import brian_no_units  # Speeds things up
import brian  # Raises a bunch of warnings, no thanks

from .dtw import dtw
from .gestures import ges_path
from . import experiments, filters, plots, processes, sermo, timit, vtl
