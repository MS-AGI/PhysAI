from .models.pinn import *
from .models.fno import *
from .models.spectral_element import *
from .models.spectralpinn import *
from .solvers import *
from .core.losses import *
from .core.auto_optimizer import *
from .core.pde_residual import *
from .trainer import *
from .visualization import *
from .visualization_nd import *
from .chat_setup import set_chat_enabled, reset_chat_consent, ensure_chat_consent
from .dedalus_setup import (
    install_dedalus,
    is_dedalus_installed,
    set_dedalus_notice_enabled,
    reset_dedalus_notice,
)
from .geometry import *
from .utils import *

# Non-blocking check for the optional Dedalus backend. 
# Never raises, never prompts/installs anything in CI/headless/cloud contexts —
# see physai/dedalus_setup.py for the full policy.
from .dedalus_setup import notify_dedalus_status as _notify_dedalus_status

_notify_dedalus_status()
del _notify_dedalus_status