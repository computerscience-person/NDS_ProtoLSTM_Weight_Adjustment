from .schemas.models import RulesUsed

def convert_dict_to_sorted_list(model_instance: RulesUsed) -> tuple[list[str], list[bool]]:
    """
    Converts a Pydantic model with a dictionary field to a list,
    sorted by the keys of the dictionary.

    Args:
        model_instance: An instance of RulesUsed.

    Returns:
        A list of dictionary keys and a list of the dictionary values, sorted according to the dictionary keys.
    """
    sorted_keys = sorted(model_instance.rules_used.keys())
    sorted_list = [model_instance.rules_used[key] for key in sorted_keys]
    return sorted_keys, sorted_list
