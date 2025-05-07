import gradio as gr
from tts_webui.utils.manage_model_state import manage_model_state, unload_model
from tts_webui.utils.list_dir_models import unload_model_button


REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"


CHECKPOINT_DIR = "data/models/ace_step/"
# CHECKPOINT_DIR = "data/models/ace_step/default/"
USE_HALF_PRECISION = False


@manage_model_state("ace_step")
def get_model(
    model_name=REPO_ID,
    use_half_precision=None,
    torch_compile=False,
    checkpoint_dir=CHECKPOINT_DIR,
):
    from acestep.pipeline_ace_step import ACEStepPipeline

    if use_half_precision is None:
        use_half_precision = USE_HALF_PRECISION

    dtype = "bfloat16" if use_half_precision else "float32"

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        torch_compile=torch_compile,
    )

    return model_demo


def switch_half_precision(use_half_precision, use_torch_compile):
    global USE_HALF_PRECISION, USE_TORCH_COMPILE
    USE_HALF_PRECISION = use_half_precision
    USE_TORCH_COMPILE = use_torch_compile
    unload_model("ace_step")
    return "Model will be reloaded when you run the next generation."


@manage_model_state("ace_step")
def get_sampler(model_name=REPO_ID):
    from acestep.data_sampler import DataSampler

    data_sampler = DataSampler()
    return data_sampler


def infer(*args, **kwargs):
    model_demo = get_model(REPO_ID)

    return model_demo(*args, **kwargs)


def sample_data(*args, **kwargs):
    data_sampler = get_sampler(REPO_ID)

    return data_sampler.sample(*args, **kwargs)


def ui():

    from acestep.ui.components import create_text2music_ui

    gr.Markdown(
        """
        <h1 style="text-align: center;">ACE-Step: A Step Towards Music Generation Foundation Model</h1>

        Note: The extension does not currently support automatic file saving.

        Weights size: 8gb. VRAM ~22gb at full, 11gb at half precision
    """
    )

    with gr.Row():
        with gr.Column():
            use_half_precision = gr.Checkbox(
                label="Use half precision",
                value=USE_HALF_PRECISION,
            )
            use_torch_compile = gr.Checkbox(
                label="Use torch compile",
                value=False,
            )
            reload_model_button = gr.Button("Apply Settings and Unload Model")
            reload_model_button.click(
                fn=lambda use_half_precision, use_torch_compile: switch_half_precision(
                    use_half_precision,
                    use_torch_compile,
                ),
                inputs=[use_half_precision, use_torch_compile],
                outputs=[reload_model_button],
                api_name="ace_step_reload_model",
            )
            with gr.Row():
                unload_model_button("ace_step")

    create_text2music_ui(
        gr=gr,
        text2music_process_func=infer,
        sample_data_func=sample_data,
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
