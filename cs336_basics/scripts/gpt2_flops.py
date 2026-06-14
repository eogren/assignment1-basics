def ask_int(prompt, default=None):
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  please enter an integer")


def attention_module_flops(context_length: int, d_model: int) -> int:
    projections = 4 * (2 * context_length * d_model * d_model)
    dot = 2 * context_length * d_model * context_length
    v = 2 * context_length * context_length * d_model

    return projections + dot + v


def swiglu_flops(context_length: int, d_model: int, d_ff: int) -> int:
    up_projects = 2 * (2 * context_length * d_model * d_ff)
    down_project = 2 * context_length * d_ff * d_model

    return up_projects + down_project


def final_output_flops(context_length: int, d_model: int, vocab_size: int) -> int:
    return 2 * context_length * d_model * vocab_size


def main():
    vocab_size = ask_int("vocab_size", 50257)
    context_length = ask_int("context_length", 1024)
    num_layers = ask_int("num_layers", 48)
    d_model = ask_int("d_model", 1600)
    d_ff = ask_int("d_ff", 4288)

    flops_per_transformer = attention_module_flops(context_length, d_model) + swiglu_flops(
        context_length, d_model, d_ff
    )

    final_flops = final_output_flops(context_length, d_model, vocab_size)

    total_transformer_flops = num_layers * flops_per_transformer
    total_flops = total_transformer_flops + final_flops

    print(
        f"{total_flops:.3e} total flops required. {total_transformer_flops:.3e} for transformers ({flops_per_transformer:.3e} per transformer), {final_flops:.3e} final flops"
    )
