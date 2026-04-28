from vllm import LLM, SamplingParams
from PIL import Image
import os
from typing import Dict

_llm = None  # singleton

def get_model():
    global _llm
    if _llm is None:
        print("🚀 Carregando modelo...")
        _llm = LLM(
            model="Qwen/Qwen2-VL-7B-Instruct",
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.88,
            trust_remote_code=True,
        )
        print("✅ Modelo pronto!")
    return _llm

def triage_inference(image_path: str, symptoms: Dict) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    symptoms_text = "\n".join([f"- {k}: {v}" for k, v in symptoms.items()])

    prompt = f"""<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Você é um assistente de triagem de saúde. Analise o espectrograma da tosse \
e os sintomas do paciente. Use os sintomas como base principal da avaliação \
e o espectrograma como contexto visual complementar.

Sintomas relatados:
{symptoms_text}

Com base nesses dados, classifique em uma das três categorias:
🟢 SAUDÁVEL — nenhuma ação necessária
🟡 ATENÇÃO — monitore em casa, beba água, descanse
🔴 PROCURE UM MÉDICO — sintomas sugerem necessidade de avaliação

Responda com: categoria + orientação prática em 2-3 frases simples.\
<|im_end|>
<|im_start|>assistant
"""

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=300,
        top_p=0.9,
        stop=["<|im_end|>"]
    )

    outputs = get_model().generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        sampling_params
    )
    return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":
    sample_image = "mel_spectrogram_exemplo.png"

    sample_symptoms = {
        "Idade": 68,
        "Temperatura (°C)": 39.0,
        "Falta de Ar": "Sim, moderada ao esforço",
        "Tosse": "Persistente há 5 dias, com catarro",
        "Dor no Peito": "Sim, ao tossir",
        "Fadiga": "Alta"
    }

    if not os.path.exists(sample_image):
        print(f"❌ Arquivo não encontrado: {sample_image}")
        print("Execute primeiro o gerador de Mel-Spectrograma.")
    else:
        print("🔍 Iniciando triagem multimodal (MI300X + vLLM)...\n")
        
        resultado = triage_inference(sample_image, sample_symptoms)
        
        print("=" * 75)
        print("📋 RESULTADO DA TRIAGEM PREVENTIVA")
        print("=" * 75)
        print(resultado)
        print("=" * 75)