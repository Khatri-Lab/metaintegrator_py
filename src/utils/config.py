from pathlib import Path
import yaml


def load_config(path: str | None) -> dict:
    """
    Loads a configuration file and parses its contents into a dictionary.

    This function reads from a file path provided as an argument and
    uses the YAML parser to safely parse its contents into a dictionary.
    If the path is not provided or is None, it returns an empty dictionary.

    Args:
        path (str | None): The file path of the configuration file to be
            loaded. If None or an empty string, an empty dictionary is
            returned.

    Returns:
        dict: A dictionary containing the parsed configuration data. If
            the file does not contain valid YAML data or is empty, this
            function returns an empty dictionary.
    """
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config(cli_args, cfg: dict) -> dict:
    """
    Merges configuration from a command-line arguments object and a dictionary-based
    configuration file. Ensures that command-line arguments take precedence over
    file-based configuration.

    Args:
        cli_args: The command-line arguments namespace containing configuration
            options provided via CLI.
        cfg (dict): The base configuration dictionary, typically parsed from a
            configuration file.

    Returns:
        dict: A dictionary containing the merged configuration, prioritizing
        command-line arguments over file configuration when conflicts occur.
    """
    cli_dict = vars(cli_args).copy()
    command = cli_dict.get("command")

    if not command:
        return cli_dict

    # Extract command-specific config from the file
    file_command_cfg = cfg.get(command, {})

    # Start with file config and update with CLI args to ensure CLI takes precedence
    final_config = file_command_cfg.copy()
    for key, value in cli_dict.items():
        if value is not None:
            final_config[key] = value

    return final_config
