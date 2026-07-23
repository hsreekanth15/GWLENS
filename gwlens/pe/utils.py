import os
import configparser



def load_lens_tables_from_ini(ini_path: str) -> dict[str, str]:
    """
    Read lens table paths from the [lensing] section of a bilby .ini file.
    Returns a dict  {lens_name: table_path}
    """
    config = configparser.ConfigParser()
    config.read(ini_path)

    if "lensing" not in config:
        raise KeyError(f"No [lensing] section found in {ini_path}")

    lens_section = config["lensing"]
    tables = {}
    for key, value in lens_section.items():
        if key.startswith("lens_table_"):
            lens_name = key.replace("lens_table_", "")   # e.g. "sis", "point_mass"
            tables[lens_name] = value
    return tables


def load_lens_tables_from_ini(ini_path: str) -> dict[str, str]:
    config = configparser.ConfigParser()
    config.read(ini_path)
    return dict(config["tables"])   # {"sis": "/path/...", "point_mass": "/path/..."}


def setup_lenses_from_ini(ini_path: str):
    """
    Register all lenses found in the .ini file into LENS_REGISTRY.
    Call this once at the top of your bilby run script.
    """
    lens_class_map = {
        "sis":         SISLens,
        "point_mass":  PointMassLens,
        "nfw":         NFWLens,        # add yours here
    }

    tables = load_lens_tables_from_ini(ini_path)
    for lens_name, table_path in tables.items():
        if lens_name not in lens_class_map:
            raise ValueError(f"Unknown lens '{lens_name}' in ini file. "
                             f"Known: {list(lens_class_map)}")
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table for '{lens_name}' not found: {table_path}")
        register_lens(lens_name, lens_class_map[lens_name](table_path=table_path))
        print(f"  Registered lens '{lens_name}' → {table_path}")