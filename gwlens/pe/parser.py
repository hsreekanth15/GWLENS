import bilby_pipe.parser
from bilby_pipe.parser import StoreBoolean
from bilby_pipe.utils import nonestr


def create_parser(top_level=True):
    """
    An ini parser for the GWLENS arguments to be used by gwlens_pipe: bilby_pipe's parser plus the
    "Lensing_arguments" group.
    """

    parser = bilby_pipe.parser.create_parser(top_level=top_level)
    group = parser.add_argument_group("Lensing_arguments", description="GWLENS lensed signal PE arguments")

    group.add("--lens-model", type=nonestr, default=None,
              help="Lens model: point, sis, cis, nfw, binary_point, point_shear, or the dotted path of your "
                   "own lens function build(m, *shape_parameters)")
    group.add("--lens-solver", type=str, default="auto", choices=["auto", "1d", "2d"],
              help="Solver: auto (the lens model's own), 1d (axisymmetric lenses) or 2d (any lens)")
    group.add("--use-lookup-table", action=StoreBoolean, default=False,
              help="If True, a lookup table will be used instead of using the solver directly")
    group.add("--lensed-lookup-generation", action=StoreBoolean, default=False,
              help="If True, a lookup table will be generated instead of the lookup-table provided")
    group.add("--lens-time-origin", type=str, default="first_image", choices=["none","first_image"])
    group.add("--lens-lookup-table", type=nonestr, default=None,
              help="Path to amplification factor lookup-table in .hdf5 (generated from the prior range if "
                   "use-lookup-table is True and the file does not exist)")
    group.add("--lens-lookup-npool", type=int, default=1,
              help="Number of processes used to generate the lookup table")
    group.add("--lens-lookup-kwargs", type=nonestr, default=None,
              help="Grid of a generated lookup table, e.g. {dw: 0.05, dy: 0.002}")
    group.add("--lensed-injection", action=StoreBoolean, default=False,
              help="If True, inject with the lensed waveform for the chosen lens model and "
                   "injection-frequency-domain-waveform and approximant. If False, it is left to default unlensed")
    group.add("--lensed-recovery", action=StoreBoolean, default=False,
              help="If True: recovery with lensed waveform, If False, frequency-domain-source-model is left as "
                   "it is (unlensed case)")

    return parser