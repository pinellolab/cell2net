from typing import Any, Literal

from cell2net._logging import logger

from captum.attr import Attribution

ATTR_METHODS = Literal[
    "deeplift",
    "integrated_gradients",
    "gradient_shap",
    "input_x_gradient",
    "input_x_gradient_times_input",
    "input_x_times_gradient",
]

def get_attr_methods(attr_method: ATTR_METHODS) -> Attribution:
    """
    Returns the appropriate attribution method from Captum based on the specified method name.

    Parameters
    ----------
    attr_method : ATTR_METHODS
        The name of the attribution method to retrieve.

    Returns
    -------
    Attribution
        The corresponding Captum attribution method class.
    """
    if attr_method == "deeplift":
        from captum.attr import DeepLift
        return DeepLift()
    elif attr_method == "integrated_gradients":
        from captum.attr import IntegratedGradients
        return IntegratedGradients()
    elif attr_method == "gradient_shap":
        from captum.attr import GradientShap
        return GradientShap()
    elif attr_method == "input_x_gradient":
        from captum.attr import InputXGradient
        return InputXGradient()
    elif attr_method == "input_x_gradient_times_input":
        from captum.attr import InputXGradientTimesInput
        return InputXGradientTimesInput()
    elif attr_method == "input_x_times_gradient":
        from captum.attr import InputXTimesGradient
        return InputXTimesGradient()

    raise ValueError(f"Unknown attribution method: {attr_method}")
