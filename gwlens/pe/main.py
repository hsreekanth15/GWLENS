"""
gwlens_pipe: top-level CLI.

Strategy
--------
gwlens_pipe only wraps the *submission* step of bilby_pipe. It

1. parses the ini with the extended parser (so lens-model and
   lens-lookup-table are first-class ini options),
2. resolves lens-model -> frequency-domain-source-model via the GWLENS
   registry,
3. resolves lens-lookup-table to an absolute path and injects it into
   waveform-arguments-dict,
4. hands everything to the unmodified bilby_pipe machinery.

Because the lensing information is baked into standard bilby_pipe options
before the DAG is written, every downstream job (bilby_pipe_generation,
bilby_pipe_analysis, ...) runs the stock executables -- nothing else in
bilby_pipe needs patching, and the run is reproducible from the
*_config_complete.ini alone.
"""

import ast
import os
import sys

from bilby_pipe.main import MainInput, generate_dag, write_complete_config_file
from bilby_pipe.utils import get_command_line_arguments, parse_args

from .waveform import LENS_MODELS, LENS_MODEL_PARAMETERS
from .parser import create_parser

try:
    from bilby_pipe.utils import logger
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger("gwlens_pipe")


def _coerce_dict(value):
    """bilby_pipe stores dict-type ini options as strings; normalise both."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from bilby_pipe.utils import convert_string_to_dict
        return convert_string_to_dict(value) or {}
    except Exception:
        return ast.literal_eval(value)


def apply_lensing_options(args):
    """Validate lensing options and fold them into standard bilby_pipe args."""
    if args.lens_model is None:
        raise ValueError(
            "gwlens_pipe requires `lens-model` to be set in the ini file. "
            "Available models: {}".format(", ".join(sorted(LENS_MODELS)))
        )
    lens_model = args.lens_model.lower()
    available_lens_models = sorted({k[0] for k in LENS_MODELS})
    if lens_model not in available_lens_models:
        raise ValueError(
            "Unknown lens-model '{}'. Available models: {}".format(
                args.lens_model, ", ".join(available_lens_models)))

    if args.lens_lookup_table is None:
        raise ValueError("gwlens_pipe requires `lens-lookup-table` to be set. Aborting gwlens-pipe..")
    
    lookup_table = os.path.abspath(os.path.expanduser(args.lens_lookup_table))
    if not os.path.isfile(lookup_table):
        raise FileNotFoundError(
            "lens-lookup-table not found: {}".format(lookup_table))
    args.lens_lookup_table = lookup_table


    # 2. Ship the lookup-table path (and the model name, for provenance)
    #    to the waveform generator
    waveform_arguments = _coerce_dict(
        getattr(args, "waveform_arguments_dict", None))
    waveform_arguments["lens_lookup_table"] = lookup_table
    waveform_arguments["lens_model"] = lens_model
    
    #---Recovery---

    base_recovery = args.frequency_domain_source_model or "lal_binary_black_hole"
    if args.lensed_recovery:
        key = (lens_model,base_recovery)
        if key not in LENS_MODELS:
            raise ValueError(
                f"No GWLENs source model for lens-model='{lens_model}'"
                f"with base '{base_recovery}'. Available :{sorted(LENS_MODELS)}"
            )
        args.frequency_domain_source_model = LENS_MODELS[key]
        logger.info("GWLENS recovery model:%s",args.frequency_domain_source_model)

    else:
        args.frequency_domain_source_model = base_recovery
        logger.info("GWLENS recovery model (unlensed):%s",base_recovery)

     # --- Injection ---
    base_injection = getattr(args, "injection_frequency_domain_source_model", None) \
        or base_recovery
    if args.lensed_injection:
        key = (lens_model, base_injection)
        if key not in LENS_MODELS:
            raise ValueError(
                f"No GWLENS source model for lens-model='{lens_model}' "
                f"with base '{base_injection}'. Available: {sorted(LENS_MODELS)}")
        args.injection_frequency_domain_source_model = LENS_MODELS[key]
        logger.info("GWLENS injection: %s", args.injection_frequency_domain_source_model)
    else:
        args.injection_frequency_domain_source_model = base_injection
        logger.info("GWLENS injection: unlensed (%s)", base_injection)
    



    args.waveform_arguments_dict = str(waveform_arguments)


    logger.info("GWLENS: amplification lookup table %s", lookup_table)
    logger.info(
        "GWLENS: ensure your prior file includes the lens parameters: %s",
        ", ".join(LENS_MODEL_PARAMETERS[lens_model]))
    return args


def main():
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)
    args = apply_lensing_options(args)

    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)
    generate_dag(inputs)

    if len(unknown_args) > 0:
        logger.warning("Unrecognized arguments: %s", unknown_args)


if __name__ == "__main__":
    sys.exit(main())