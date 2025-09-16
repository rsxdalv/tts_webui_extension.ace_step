import gradio as gr
from tts_webui.utils.manage_model_state import manage_model_state, is_model_loaded
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
import functools

REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"


CHECKPOINT_DIR = "data/models/ace_step/"
USE_HALF_PRECISION = True
USE_TORCH_COMPILE = False
USE_CPU_OFFLOAD = True
USE_OVERLAPPED_DECODE = True


@manage_model_state("ace_step")
def get_model(
    model_name=REPO_ID,
    use_half_precision=None,
    torch_compile=None,
    checkpoint_dir=CHECKPOINT_DIR,
    cpu_offload=None,
    overlapped_decode=None,
):
    from acestep.pipeline_ace_step import ACEStepPipeline

    if use_half_precision is None:
        use_half_precision = USE_HALF_PRECISION

    if torch_compile is None:
        torch_compile = USE_TORCH_COMPILE

    if cpu_offload is None:
        cpu_offload = USE_CPU_OFFLOAD

    if overlapped_decode is None:
        overlapped_decode = USE_OVERLAPPED_DECODE

    dtype = "bfloat16" if use_half_precision else "float32"

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )

    return model_demo


def store_global_settings(
    use_half_precision, use_torch_compile, use_cpu_offload, use_overlapped_decode
):
    global USE_HALF_PRECISION, USE_TORCH_COMPILE, USE_CPU_OFFLOAD, USE_OVERLAPPED_DECODE
    USE_HALF_PRECISION = use_half_precision
    USE_TORCH_COMPILE = use_torch_compile
    USE_CPU_OFFLOAD = use_cpu_offload
    USE_OVERLAPPED_DECODE = use_overlapped_decode
    if is_model_loaded("ace_step"):
        return "Please unload the model to apply changes."
    return "Settings applied."


@manage_model_state("ace_step_sampler")
def get_sampler(model_name=REPO_ID):
    from acestep.data_sampler import DataSampler

    data_sampler = DataSampler()
    return data_sampler


def decorator_ace_step_adapter(fn):
    def ace_step_infer(
        format: str = "wav",
        audio_duration: float = 60.0,
        prompt: str = None,
        negative_prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str = None,
        lora_name_or_path: str = "none",
        retake_seeds: list = None,
        retake_variance: float = 0.5,
        task: str = "text2music",
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        save_path: str = None,
        batch_size: int = 1,
        debug: bool = False,
        **kwargs,
    ):
        params = locals()
        del params["kwargs"]
        del params["fn"]
        return fn(**params, text=prompt[:20])

    return ace_step_infer


# This decorator will convert the return value from a list to a dict
def to_dict_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, list) and len(result) > 0:
            import soundfile as sf

            audio, sample_rate = sf.read(result[0])

            return {"audio_out": (sample_rate, audio), "_original_result": result}
        return result

    return wrapper


# This decorator will convert the return value back from a dict to the original list
def from_dict_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        # Check if the result is a dict with the _original_result key
        if isinstance(result, dict) and "_original_result" in result:
            return result["_original_result"]
        return result

    return wrapper


# @functools.wraps(ace_step_infer)
@decorator_ace_step_adapter
@from_dict_decorator  # This will run last, converting dict back to list
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("ace_step")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@to_dict_decorator  # This will run first, converting list to dict
@log_function_time
def ace_step_infer_decorated(*args, _type, text, **kwargs):
    from acestep.pipeline_ace_step import ACEStepPipeline

    model_demo: ACEStepPipeline = get_model(REPO_ID)

    return model_demo(*args, **kwargs)


def sample_data(*args, **kwargs):
    data_sampler = get_sampler(REPO_ID)

    return data_sampler.sample(*args, **kwargs)


def load_data(*args, **kwargs):
    data_sampler = get_sampler(REPO_ID)

    return data_sampler.load_data(*args, **kwargs)


def ui():
    # from acestep.ui.components import create_text2music_ui

    from .components import create_text2music_ui

    gr.Markdown(
        """<h2 style="text-align: center;">ACE-Step: A Step Towards Music Generation Foundation Model</h2>"""
    )

    with gr.Column(variant="panel"):
        gr.Markdown("### Model Settings")
        with gr.Row():
            use_half_precision = gr.Checkbox(
                label="Use half precision",
                value=USE_HALF_PRECISION,
            )
            use_torch_compile = gr.Checkbox(
                label="Use torch compile",
                value=USE_TORCH_COMPILE,
            )
            use_cpu_offload = gr.Checkbox(
                label="Use CPU offload",
                value=USE_CPU_OFFLOAD,
            )
            use_overlapped_decode = gr.Checkbox(
                label="Use overlapped decode",
                value=USE_OVERLAPPED_DECODE,
            )
            # save global changes when any of these change:
            unload_model_button("ace_step")

        model_info = gr.Markdown("")

    for i in [
        use_half_precision,
        use_torch_compile,
        use_cpu_offload,
        use_overlapped_decode,
    ]:
        i.change(
            fn=store_global_settings,
            inputs=[
                use_half_precision,
                use_torch_compile,
                use_cpu_offload,
                use_overlapped_decode,
            ],
            outputs=[model_info],
            api_name="ace_step_reload_model",
        )

    create_text2music_ui(
        gr=gr,
        text2music_process_func=ace_step_infer_decorated,
        sample_data_func=sample_data,
        load_data_func=load_data,
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
