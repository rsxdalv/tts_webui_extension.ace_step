import gradio as gr


def extension__tts_generation_webui():
    ui_wrapper()
    return {
        "package_name": "extension_ace_step",
        "name": "ACE-Step",
        "version": "0.3.0",
        "requirements": "git+https://github.com/rsxdalv/extension_ace_step@main",
        "description": "ACE-Step: A Step Towards Music Generation Foundation Model",
        "extension_type": "interface",
        "extension_class": "audio-music-generation",
        "author": "ACE-Step",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/ACE-Step/ACE-Step",
        "extension_website": "https://github.com/rsxdalv/extension_ace_step",
        "extension_platform_version": "0.0.1",
    }


def ui_wrapper():
    from .gradio_app import ui

    ui()


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        extension__tts_generation_webui()
    demo.launch()
