"""
gwlens_pipe: top-level CLI.

gwlens_pipe wraps the *submission* step of bilby_pipe. It parses the ini with the extended parser,
and if lensed-injection and/or lensed-recovery is True it

1. reads the lens-parameter ranges from the prior (and the injection),
2. uses the lookup table, or generates it from those ranges (use-lookup-table = True),
3. passes the lensing settings to the waveform generator through waveform-arguments-dict, and
4. sets waveform-generator (recovery) and/or injection-waveform-generator (injection) to
   gwlens.pe.waveform.LensedWaveformGenerator, which lenses the unlensed frequency-domain source model
   (frequency-domain-source-model / injection-frequency-domain-source-model, with their approximants).

Everything else is stock bilby_pipe; the run is reproducible from the *_config_complete.ini.
"""
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
    """bilby_pipe stores dict-type ini options as strings; normalise both."""
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
    """The lookup table to use: lens-lookup-table if it exists (and generation is not requested),
    otherwise a table generated from the prior ranges."""
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
    """Validate the lensing options and fold them into standard bilby_pipe args."""
    if not (args.lensed_injection or args.lensed_recovery):
        if args.lens_model is not None:
            logger.warning("GWLENS: lensed-injection and lensed-recovery are False: running unlensed")
        return args
    if args.lens_model is None:
        raise ValueError("gwlens_pipe requires `lens-model` when lensed-injection or lensed-recovery is True")
    model = get_lens_model(args.lens_model, args.lens_solver)

    # 1. ranges of the lens parameters (prior, and injection values for a lensed injection)
    priors = read_prior_strings(args)
    if args.lensed_recovery:
        missing = [p for p in model.parameters if p not in priors]
        if missing:
            raise ValueError(f"GWLENS: the prior must contain the lens parameters {missing} of '{model.name}' "
                             "(fixed values are fine)")
    injection = read_injection_values(args) if args.lensed_injection else {}
    ranges = lens_parameter_ranges(model, priors=priors, frequency_band=frequency_band(args), injection=injection)

    # 2. lensing settings for the waveform generator
    waveform_arguments = _coerce_dict(getattr(args, "waveform_arguments_dict", None))
    for key in [k for k in waveform_arguments if str(k).startswith("lens_")]:
        waveform_arguments.pop(key)
    waveform_arguments.update(lens_model=model.name, lens_solver=model.solver,
                              lens_use_lookup_table=bool(args.use_lookup_table), lens_ranges=ranges)
    if args.use_lookup_table:
        args.lens_lookup_table = lookup_table_path(args, model, ranges)
        waveform_arguments["lens_lookup_table"] = args.lens_lookup_table
    args.waveform_arguments_dict = str(waveform_arguments)

    # 3. waveform generators. The source models stay unlensed; LensedWaveformGenerator lenses them.
    unlensed = args.waveform_generator if args.waveform_generator not in (None, LENSED_GENERATOR) else UNLENSED_GENERATOR
    args.frequency_domain_source_model = args.frequency_domain_source_model or "lal_binary_black_hole"
    args.injection_frequency_domain_source_model = (args.injection_frequency_domain_source_model
                                                    or args.frequency_domain_source_model)
    args.waveform_generator = LENSED_GENERATOR if args.lensed_recovery else unlensed
    # bilby_pipe reuses waveform-generator for the injection unless injection-waveform-generator is set
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