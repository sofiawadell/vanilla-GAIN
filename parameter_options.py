from typing import Any, Dict


parameter_options: Dict[str, Dict[str, Any]] = {
    "mushroom": {
        1: { "batch_size": 256,
            "hint_rate": 0.1,
            "alpha": 0.5,
            "iterations": 10000},
        2: { "batch_size": 64,
            "hint_rate": 0.1,
            "alpha": 10,
            "iterations": 10000},
        },
    "letter": {
        1: { "batch_size": 128,
            "hint_rate": 0.1,
            "alpha": 10,
            "iterations": 10000},
        2: { "batch_size": 256,
            "hint_rate": 0.5,
            "alpha": 10,
            "iterations": 10000},
        }, 
    "bank": {
        1: { "batch_size": 128,
            "hint_rate": 0.1,
            "alpha": 0.5,
            "iterations": 10000},
        2: { "batch_size": 64,
            "hint_rate": 0.5,
            "alpha": 10,
            "iterations": 10000},
        },
    "credit": {
        1: { "batch_size": 256,
            "hint_rate": 0.1,
            "alpha": 10,
            "iterations": 10000},
        2: { "batch_size": 64,
            "hint_rate": 0.1,
            "alpha": 10,
            "iterations": 10000},
        },
}