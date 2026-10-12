
import ast
import os
import sys

from bilby_pipe.main import MainInput, generate_dag, write_complete_config_file
from bilby_pipe.utils import get_command_line_arguments, parse_args

from .parser import create_parser
from .utils import (check_lookup_table, frequency_band, generate_lookup_table, lens_parameter_ranges,
                    parse_dict_option, read_injection_values, read_prior_strings)
from .waveform import get_lens_model

try:
    from bilby_pipe.utils import logger
except ImportError:
    import logging
    logger = logging.getLogger("gwlens_pipe")

LENSED_GENERATOR = "gwlens.pe.waveform.LensedWaveformGenerator"
UNLENSED_GENERATOR = "bilby.gw.waveform_generator.LALCBCWaveformGenerator"


def _coerce_dict(value):
    """
    Bilby_pipe stores dict-type ini options as strings; normalise both.
    """
    if value is None or value == "" or value == "None":
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from bilby_pipe.utils import convert_string_to_dict
        return convert_string_to_dict(value) or {}
    except Exception:
        return ast.literal_eval(value)


def lookup_table_path(args, model, ranges):
    """

    The function checks if --lens_lookup_table=path/to/lookup.hdf5 is given.
    it takes the path and check for problems in the lookup table with the function
    "utils.check_lookup_table()" and prints the warning if there are any.

    It will generate a lookup table using generate_lookup_table() function in utils.
    
    """
    given = os.path.abspath(os.path.expanduser(args.lens_lookup_table)) if args.lens_lookup_table else None
    if given and os.path.isfile(given) and not args.lensed_lookup_generation:
        for problem in check_lookup_table(given, model, ranges):
            logger.warning(f"GWLENS lookup table {given}: {problem}")
        return given
    if given and not os.path.isfile(given):
        path = given                                   # generate it where it was asked for
    else:                                              # never overwrite a table that was given
        path = os.path.join(os.path.abspath(args.outdir), "gwlens",
                            f"{args.label}_{model.name.replace('.', '_')}_lookup.hdf5")
    return generate_lookup_table(model, ranges, path, n_workers=args.lens_lookup_npool,
                                 **parse_dict_option(args.lens_lookup_kwargs))


def apply_lensing_options(args):

    """
    The function reads the arguments provided by the user in the  gwlens_pipe .ini file 
    and make it compatible with bilby_pipe args. It also checks if all required arguments are provided for PE
    
    It does the following:
    1. ranges of the lens parameters (prior, and injection values for a lensed injection)
    2. lensing settings for the waveform generator
    3. Choosing waveform generators. The source models stay unlensed LensedWaveformGenerator lenses them.
    4. Logging the details with bilby_pipe logger
    
    """
    if not (args.lensed_injection or args.lensed_recovery):
        if args.lens_model is not None:
            logger.warning("GWLENS: lensed-injection and lensed-recovery are False: running unlensed")
        return args
    if args.lens_model is None:
        raise ValueError("gwlens_pipe requires `lens-model` when lensed-injection or lensed-recovery is True")
    model = get_lens_model(args.lens_model, args.lens_solver)


    priors = read_prior_strings(args)
    if args.lensed_recovery:
        missing = [p for p in model.parameters if p not in priors]
        if missing:
            raise ValueError(f"GWLENS: the prior must contain the lens parameters {missing} of '{model.name}' "
                             "(fixed values are fine)")
    injection = read_injection_values(args) if args.lensed_injection else {}
    ranges = lens_parameter_ranges(model, priors=priors, frequency_band=frequency_band(args), injection=injection)

    
    waveform_arguments = _coerce_dict(getattr(args, "waveform_arguments_dict", None))
    for key in [k for k in waveform_arguments if str(k).startswith("lens_")]:
        waveform_arguments.pop(key)
    waveform_arguments.update(lens_model=model.name, lens_solver=model.solver,
                              lens_use_lookup_table=bool(args.use_lookup_table), lens_ranges=ranges)
    if args.use_lookup_table:
        args.lens_lookup_table = lookup_table_path(args, model, ranges)
        waveform_arguments["lens_lookup_table"] = args.lens_lookup_table
    args.waveform_arguments_dict = str(waveform_arguments)

    
    unlensed = args.waveform_generator if args.waveform_generator not in (None, LENSED_GENERATOR) else UNLENSED_GENERATOR
    args.frequency_domain_source_model = args.frequency_domain_source_model or "lal_binary_black_hole"
    args.injection_frequency_domain_source_model = (args.injection_frequency_domain_source_model
                                                    or args.frequency_domain_source_model)
    args.waveform_generator = LENSED_GENERATOR if args.lensed_recovery else unlensed
    
    args.injection_waveform_generator = LENSED_GENERATOR if args.lensed_injection else unlensed

    approximant = args.injection_waveform_approximant or args.waveform_approximant
    logger.info(f"GWLENS: lens model '{model.name}' ({model.solver} solver, parameters "
                f"{', '.join(model.parameters)}), F from {'lookup table' if args.use_lookup_table else 'the solver'}")
    logger.info(f"GWLENS recovery: {'lensed' if args.lensed_recovery else 'unlensed'} "
                f"{args.frequency_domain_source_model} ({args.waveform_approximant})")
    logger.info(f"GWLENS injection: {'lensed' if args.lensed_injection else 'unlensed'} "
                f"{args.injection_frequency_domain_source_model} ({approximant})")
    logger.info(f"GWLENS: lens-parameter ranges {ranges}")
    return args


def main():
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)
    args.outdir = args.outdir.replace("'", "").replace('"', "")
    if args.overwrite_outdir is False:
        from bilby_pipe.utils import get_outdir_name
        args.outdir = get_outdir_name(args.outdir)
    args = apply_lensing_options(args)

    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)
    generate_dag(inputs)

    if len(unknown_args) > 0:
        logger.warning("Unrecognized arguments: %s", unknown_args)


if __name__ == "__main__":
    sys.exit(main())