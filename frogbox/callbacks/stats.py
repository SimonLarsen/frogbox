from math import floor, log

from frogbox.pipelines.pipeline import Pipeline

UNITS = ["", "K", "M", "B"]


def print_model_parameters(
    pipe: Pipeline,
    model_key: str = "model",
    short_form: bool = True,
) -> None:
    """
    Print number of parameters for model with key `model_key`.

    Parameters
    ----------
    model_key
        Model key to print stats for.
    short_form
        If true numbers will be formatted as short form e.g. "23.16 M".
    """
    model = pipe._models[model_key]
    num_params = sum(p.numel() for p in model.parameters())

    mag = floor(log(num_params, 1e3))
    mag = min(mag, len(UNITS) - 1)

    if mag > 0 and short_form:
        num_str = f"{num_params / 1e3**mag:.2f} {UNITS[mag]}"
    else:
        num_str = str(num_params)
    msg = f"{model_key} parameters: {num_str}"
    pipe.print(msg)
