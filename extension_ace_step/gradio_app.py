import gradio as gr
from tts_webui.utils.manage_model_state import manage_model_state, unload_model
from tts_webui.utils.list_dir_models import unload_model_button


REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"


CHECKPOINT_DIR = "data/models/ace_step/"
# CHECKPOINT_DIR = "data/models/ace_step/default/"
USE_HALF_PRECISION = False
USE_TORCH_COMPILE = False
USE_CPU_OFFLOAD = False


@manage_model_state("ace_step")
def get_model(
    model_name=REPO_ID,
    use_half_precision=None,
    torch_compile=None,
    checkpoint_dir=CHECKPOINT_DIR,
    cpu_offload=None,
):
    from acestep.pipeline_ace_step import ACEStepPipeline

    if use_half_precision is None:
        use_half_precision = USE_HALF_PRECISION

    if torch_compile is None:
        torch_compile = USE_TORCH_COMPILE

    if cpu_offload is None:
        cpu_offload = USE_CPU_OFFLOAD

    # if use_half_precision:
    #     cpu_offload = True
    # else:
    #     cpu_offload = False

    dtype = "bfloat16" if use_half_precision else "float32"

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
    )

    return model_demo


def switch_half_precision(use_half_precision, use_torch_compile, use_cpu_offload):
    global USE_HALF_PRECISION, USE_TORCH_COMPILE, USE_CPU_OFFLOAD
    USE_HALF_PRECISION = use_half_precision
    USE_TORCH_COMPILE = use_torch_compile
    USE_CPU_OFFLOAD = use_cpu_offload
    unload_model("ace_step")
    return "Model will be reloaded when you run the next generation."


@manage_model_state("ace_step")
def get_sampler(model_name=REPO_ID):
    from acestep.data_sampler import DataSampler

    data_sampler = DataSampler()
    return data_sampler


# def ace_step_infer(*args, ref_audio_input=None, **kwargs):
#     model_demo = get_model(REPO_ID)

#     # return model_demo(*args, **kwargs)
#     # return model_demo(*args, ref_audio_input=ref_audio_input, **kwargs)
#     # TypeError: ACEStepPipeline.__call__() got multiple values for argument 'ref_audio_input'
#     return model_demo(*args, ref_audio_input=ref_audio_input, **kwargs)


def ace_step_infer(
    audio_duration: float = 60.0,
    prompt: str = None,
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
    format: str = "wav",
    batch_size: int = 1,
    debug: bool = False,
):
    model_demo = get_model(REPO_ID)

    return model_demo(
        audio_duration=audio_duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=omega_scale,
        manual_seeds=manual_seeds,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
        min_guidance_scale=min_guidance_scale,
        use_erg_tag=use_erg_tag,
        use_erg_lyric=use_erg_lyric,
        use_erg_diffusion=use_erg_diffusion,
        oss_steps=oss_steps,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        audio2audio_enable=audio2audio_enable,
        ref_audio_strength=ref_audio_strength,
        ref_audio_input=ref_audio_input,
        retake_seeds=retake_seeds,
        retake_variance=retake_variance,
        task=task,
        repaint_start=repaint_start,
        repaint_end=repaint_end,
        src_audio_path=src_audio_path,
        edit_target_prompt=edit_target_prompt,
        edit_target_lyrics=edit_target_lyrics,
        edit_n_min=edit_n_min,
        edit_n_max=edit_n_max,
        edit_n_avg=edit_n_avg,
        save_path=save_path,
        format=format,
        batch_size=batch_size,
        debug=debug,
    )


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

        CPU Offload can be enabled to reduce VRAM usage, but it does slow down the generation process, especially for short generations.
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
            use_cpu_offload = gr.Checkbox(
                label="Use CPU offload",
                value=False,
            )
            reload_model_button = gr.Button("Apply Settings and Unload Model")
            reload_model_button.click(
                fn=switch_half_precision,
                inputs=[use_half_precision, use_torch_compile, use_cpu_offload],
                outputs=[reload_model_button],
                api_name="ace_step_reload_model",
            )
            with gr.Row():
                unload_model_button("ace_step")

    create_text2music_ui(
        gr=gr,
        text2music_process_func=ace_step_infer,
        sample_data_func=sample_data,
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
