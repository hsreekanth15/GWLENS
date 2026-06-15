import bilby_pipe.parser


def create_parser(top_level=True):
    """
    Creating and defining a parser for the lens arguments
    """

    parser=bilby_pipe.parser.create_parser(top_level=top_level)
    group = parser.add_argument_group("Lensing_arguments", description="GWLENS lensed signal PE arguments")

    group.add("--lens-model", type=str,default=None,help=("Name of the lens model used in the analysis for GWLENS"))
    group.add("--lens-lookup-table",type=str,default=None,help=("Path to amplification factor lookup-table in .hdf5"))
    group.add("--lensed-injection", action="store_true",default=True, help="If True, inject with the lensed waveform for the chosen lens model and injection-frequency-domain-waveform and approximant. If False, it is left to default unlensed")
    group.add("--lensed-recovery",action="store_true",default=True, help= "If True: recovery with lensed waveform, If False, frequency-domain-source-model is left as it is (unlensed case)")


    return parser



