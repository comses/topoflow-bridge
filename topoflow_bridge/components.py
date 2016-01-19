__all__ = []


import sys
import importlib
import warnings
import inspect

from bmi_bridge import bmi_factory
from topoflow.utils.BMI_base import BMI_component


TOPOFLOW_COMPONENTS = [
    ('channels_diffusive_wave', 'channels_component'),
    ('channels_dynamic_wave', 'channels_component'),
    ('channels_kinematic_wave', 'channels_component'),
    ('d8_global', 'd8_component'),
    ('d8_local', 'd8_component'),
    ('diversions_fraction_method', 'diversions_component'),
    ('erode_d8_global', 'erosion_component'),
    ('erode_d8_local', 'erosion_component'),
    ('evap_energy_balance', 'evap_component'),
    ('evap_priestley_taylor', 'evap_component'),
    ('evap_read_file', 'evap_component'),
    ('infil_beven', 'infil_beven'),
    ('infil_green_ampt', 'infil_component'),
    ('infil_richards_1D', 'infil_component'),
    ('infil_smith_parlange', 'infil_component'),
    ('met_base', 'met_component'),
    ('satzone_darcy_layers', 'satzone_component'),
    ('snow_degree_day', 'snow_component'),
    ('snow_energy_balance', 'snow_component'),
]


def import_topoflow_component(mod_name, cls_name):
    try:
        topoflow_module = importlib.import_module(
            '.'.join(['topoflow', 'components', mod_name]))
    except ImportError:
        warnings.warn('unable to import {mod}'.format(mod=mod_name))
    else:
        obj = topoflow_module.__dict__[cls_name]
        bmi_obj = bmi_factory(obj, name=mod_name)
        setattr(sys.modules[__name__], bmi_obj.__name__, bmi_obj)
        __all__.append(bmi_obj.__name__)


for name in TOPOFLOW_COMPONENTS:
    import_topoflow_component(*name)
