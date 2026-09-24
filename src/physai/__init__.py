from .models.pinn import *
from .models.fno import *
from .models.spectral_element import *
from .models.spectralpinn import *
from .solvers import *
from .core.losses import *
from .core.auto_optimizer import *
from .core.pde_residual import *
from .core.latex_pde import *
from .trainer import *
from .visualization import *
from .visualization_nd import *
from .chat_setup import set_chat_enabled, reset_chat_consent, ensure_chat_consent
from .solver_setup import (
    install_solver_dependencies,
    missing_solver_dependencies,
    is_dedalus_installed,
    set_solver_notice_enabled,
    reset_solver_notice,
    install_dedalus,
    set_dedalus_notice_enabled,
    reset_dedalus_notice,
)
from .geometry import *
from .utils import *

# Non-blocking check for optional classical solver dependencies.
# Never raises, never prompts/installs anything in CI/headless/cloud contexts —
# see physai/solver_setup.py for the full policy.
from .solver_setup import notify_solver_status as _notify_solver_status

_notify_solver_status()
del _notify_solver_status
