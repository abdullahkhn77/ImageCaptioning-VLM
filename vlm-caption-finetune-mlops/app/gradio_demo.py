"""
Gradio demo for VLM image captioning.
Run: python app/gradio_demo.py
"""

import os
from pathlib import Path

import gradio as gr


def get_predictor():
    """Load predictor from env or default path."""
    ckpt = os.environ.get("VLM_CHECKPOINT_PATH", "outputs/checkpoints")
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt}. Set VLM_CHECKPOINT_PATH to a trained checkpoint."
        )
    from vlm_caption_finetune_mlops.inference import load_predictor
    return load_predictor(ckpt)


def main():
    predictor = get_predictor()

    def caption_fn(image):
        """Generate caption for a single image (PIL or numpy)."""
        if image is None:
            return ""
        return predictor.predict(image)

    demo = gr.Interface(
        fn=caption_fn,
        inputs=gr.Image(type="pil", label="Upload image"),
        outputs=gr.Textbox(label="Caption"),
        title="VLM Image Captioning",
        description="Upload an image to get a caption from a fine-tuned Vision-Language Model.",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
