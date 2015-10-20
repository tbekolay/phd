import warnings
warnings.simplefilter("ignore")  # We don't care!
import brian_no_units  # Speeds things up
import brian  # Raises a bunch of warnings, no thanks
warnings.simplefilter("default")

from .sermo import Sermo
from . import filters, plots, sounds
