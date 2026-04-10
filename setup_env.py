import argparse
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("setup_env")

SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": {
        "model_name": "bitnet_b1_58-large",
    },
    "1bitLLM/bitnet_b1_58-3B": {
        "model_name": "bitnet_b1_58-3B",
    },
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {
        "model_name": "Llama3-8B-1.58-100B-tokens",
    },
    "tiiuae/Falcon3-7B-Instruct-1.58bit": {
        "model_name": "Falcon3-7B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-7B-1.58bit": {
        "model_name": "Falcon3-7B-1.58bit",
    },
    "tiiuae/Falcon3-10B-Instruct-1.58bit": {
        "model_name": "Falcon3-10B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-10B-1.58bit": {
        "model_name": "Falcon3-10B-1.58bit",
    },
    "tiiuae/Falcon3-3B-Instruct-1.58bit": {
        "model_name": "Falcon3-3B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-3B-1.58bit": {
        "model_name": "Falcon3-3B-1.58bit",
    },
    "tiiuae/Falcon3-1B-Instruct-1.58bit": {
        "model_name": "Falcon3-1B-Instruct-1.58bit",
    },
    "microsoft/BitNet-b1.58-2B-4T": {
        "model_name": "BitNet-b1.58-2B-4T",
    },
    "tiiuae/Falcon-E-3B-Instruct": {
        "model_name": "Falcon-E-3B-Instruct",
    },
    "tiiuae/Falcon-E-1B-Instruct": {
        "model_name": "Falcon-E-1B-Instruct",
    },
    "tiiuae/Falcon-E-3B-Base": {
        "model_name": "Falcon-E-3B-Base",
    },
    "tiiuae/Falcon-E-1B-Base": {
        "model_name": "Falcon-E-1B-Base",
    },
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"]
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}

WINDOWS_CMAKE_GENERATORS = {
    "18": "Visual Studio 18 2026",
    "17": "Visual Studio 17 2022",
    "16": "Visual Studio 16 2019",
}

CMAKE_ARCHITECTURES = {
    "arm64": "ARM64",
    "x86_64": "x64",
}

def system_info():
    machine = platform.machine()
    if machine not in ARCH_ALIAS:
        logging.error(f"Arch {machine} is not supported yet")
        sys.exit(1)
    return platform.system(), ARCH_ALIAS[machine]

def get_model_name():
    if args.hf_repo:
        return SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]
    return os.path.basename(os.path.normpath(args.model_dir))

def get_model_dir():
    if args.hf_repo:
        return os.path.join(args.model_dir, SUPPORTED_HF_MODELS[args.hf_repo]["model_name"])
    return args.model_dir

def get_gguf_path():
    return os.path.join(get_model_dir(), "ggml-model-" + args.quant_type + ".gguf")

def run_command(command, shell=False, log_step=None):
    """Run a system command and ensure it succeeds."""
    try:
        if log_step:
            log_file = os.path.join(args.log_dir, log_step + ".log")
            with open(log_file, "w") as f:
                subprocess.run(command, shell=shell, check=True, stdout=f, stderr=subprocess.STDOUT)
            return
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        if log_step:
            logging.error(f"Error occurred while running command: {e}, check details in {log_file}")
        else:
            logging.error(f"Error occurred while running command: {e}")
        sys.exit(1)

def find_hf_cli():
    for candidate in ("hf", "huggingface-cli"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    logging.error("Neither `hf` nor `huggingface-cli` was found in PATH.")
    sys.exit(1)

def get_vs_installation():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    try:
        result = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-format", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    installations = json.loads(result.stdout)
    if not installations:
        return None
    return installations[0]

def find_cmake():
    cmake = shutil.which("cmake")
    if cmake is not None:
        return cmake
    if platform.system() == "Windows":
        vs_installation = get_vs_installation()
        if vs_installation is not None:
            candidate = (
                Path(vs_installation["installationPath"])
                / "Common7"
                / "IDE"
                / "CommonExtensions"
                / "Microsoft"
                / "CMake"
                / "CMake"
                / "bin"
                / "cmake.exe"
            )
            if candidate.exists():
                return str(candidate)
    logging.error("CMake is not available. Please install it or use a Visual Studio installation that bundles CMake.")
    sys.exit(1)

def get_windows_generator():
    vs_installation = get_vs_installation()
    if vs_installation is not None:
        major_version = vs_installation.get("installationVersion", "").split(".", 1)[0]
        generator = WINDOWS_CMAKE_GENERATORS.get(major_version)
        if generator is not None:
            return generator
    return WINDOWS_CMAKE_GENERATORS["18"]

def get_cmake_args(arch):
    if arch == "arm64":
        return [f"-DBITNET_ARM_TL1={'ON' if args.quant_type == 'tl1' else 'OFF'}"]
    if arch == "x86_64":
        return [f"-DBITNET_X86_TL2={'ON' if args.quant_type == 'tl2' else 'OFF'}"]
    logging.error(f"Arch {arch} is not supported yet")
    sys.exit(1)

def get_binary_path(name):
    suffixes = [".exe", ""] if platform.system() == "Windows" else [""]
    prefixes = [
        Path("build") / "bin" / "Release",
        Path("build") / "bin",
    ]
    for prefix in prefixes:
        for suffix in suffixes:
            candidate = prefix / f"{name}{suffix}"
            if candidate.exists():
                return str(candidate)
    logging.error(f"Required binary {name} was not found under build/bin.")
    sys.exit(1)

def requires_conversion():
    gguf_path = get_gguf_path()
    return not os.path.exists(gguf_path) or os.path.getsize(gguf_path) == 0

def ensure_conversion_dependencies():
    missing = []
    for module_name in ("numpy", "torch", "gguf"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        logging.error(
            "Missing Python packages required for HF-to-GGUF conversion: %s. "
            "Install them in the active environment or use a pre-quantized GGUF model directory.",
            ", ".join(missing),
        )
        sys.exit(1)

def prepare_model():
    _, arch = system_info()
    hf_url = args.hf_repo
    model_dir = get_model_dir()
    quant_type = args.quant_type
    quant_embd = args.quant_embd
    if hf_url is not None:
        # download the model
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model {hf_url} from HuggingFace to {model_dir}...")
        run_command([find_hf_cli(), "download", hf_url, "--local-dir", model_dir], log_step="download_model")
    elif not os.path.exists(model_dir):
        logging.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)
    else:
        logging.info(f"Loading model from directory {model_dir}.")
    gguf_path = get_gguf_path()
    if not os.path.exists(gguf_path) or os.path.getsize(gguf_path) == 0:
        ensure_conversion_dependencies()
        logging.info(f"Converting HF model to GGUF format...")
        if quant_type.startswith("tl"):
            run_command([sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", quant_type, "--quant-embd"], log_step="convert_to_tl")
        else: # i2s
            # convert to f32
            run_command([sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", "f32"], log_step="convert_to_f32_gguf")
            f32_model = os.path.join(model_dir, "ggml-model-f32.gguf")
            i2s_model = os.path.join(model_dir, "ggml-model-i2_s.gguf")
            quantize_binary = get_binary_path("llama-quantize")
            # quantize to i2s
            if quant_embd:
                run_command([quantize_binary, "--token-embedding-type", "f16", f32_model, i2s_model, "I2_S", "1", "1"], log_step="quantize_to_i2s")
            else:
                run_command([quantize_binary, f32_model, i2s_model, "I2_S", "1"], log_step="quantize_to_i2s")

        logging.info(f"GGUF model saved at {gguf_path}")
    else:
        logging.info(f"GGUF model already exists at {gguf_path}")

def setup_gguf():
    if requires_conversion():
        ensure_conversion_dependencies()
    else:
        logging.info("Skipping GGUF conversion dependency check because a pre-quantized GGUF model was provided.")

def gen_code():
    _, arch = system_info()
    
    llama3_f3_models = set([model['model_name'] for model in SUPPORTED_HF_MODELS.values() if model['model_name'].startswith("Falcon") or model['model_name'].startswith("Llama")])

    if arch == "arm64":
        if args.use_pretuned:
            pretuned_kernels = os.path.join("preset_kernels", get_model_name())
            if not os.path.exists(pretuned_kernels):
                logging.error(f"Pretuned kernels not found for model {args.hf_repo}")
                sys.exit(1)
            if args.quant_type == "tl1":
                shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl1.h"), "include/bitnet-lut-kernels.h")
                shutil.copyfile(os.path.join(pretuned_kernels, "kernel_config_tl1.ini"), "include/kernel_config.ini")
            elif args.quant_type == "tl2":
                shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl2.h"), "include/bitnet-lut-kernels.h")
                shutil.copyfile(os.path.join(pretuned_kernels, "kernel_config_tl2.ini"), "include/kernel_config.ini")
        if get_model_name() == "bitnet_b1_58-large":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "128,64,128", "--bm", "32,64,32"], log_step="codegen")
        elif get_model_name() in llama3_f3_models:
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "Llama3-8B-1.58-100B-tokens", "--BM", "256,128,256,128", "--BK", "128,64,128,64", "--bm", "32,64,32,64"], log_step="codegen")
        elif get_model_name() == "bitnet_b1_58-3B":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "64,128,64", "--bm", "32,64,32"], log_step="codegen")
        elif get_model_name() == "BitNet-b1.58-2B-4T":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "64,128,64", "--bm", "32,64,32"], log_step="codegen")
        else:
            raise NotImplementedError()
    else:
        if args.use_pretuned:
            # cp preset_kernels/model_name/bitnet-lut-kernels_tl1.h to include/bitnet-lut-kernels.h
            pretuned_kernels = os.path.join("preset_kernels", get_model_name())
            if not os.path.exists(pretuned_kernels):
                logging.error(f"Pretuned kernels not found for model {args.hf_repo}")
                sys.exit(1)
            shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl2.h"), "include/bitnet-lut-kernels.h")
        if get_model_name() == "bitnet_b1_58-large":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "96,192,96", "--bm", "32,32,32"], log_step="codegen")
        elif get_model_name() in llama3_f3_models:
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "Llama3-8B-1.58-100B-tokens", "--BM", "256,128,256,128", "--BK", "96,96,96,96", "--bm", "32,32,32,32"], log_step="codegen")
        elif get_model_name() == "bitnet_b1_58-3B":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32"], log_step="codegen")
        elif get_model_name() == "BitNet-b1.58-2B-4T":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32"], log_step="codegen")    
        else:
            raise NotImplementedError()


def compile():
    cmake = find_cmake()
    _, arch = system_info()
    cmake_args = get_cmake_args(arch)
    logging.info("Compiling the code using CMake.")
    configure_command = [cmake, "--fresh", "-S", ".", "-B", "build", *cmake_args]
    if platform.system() == "Windows":
        configure_command.extend(["-G", get_windows_generator(), "-A", CMAKE_ARCHITECTURES[arch], "-T", "ClangCL"])
    else:
        configure_command.extend(["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"])
    run_command(configure_command, log_step="generate_build_files")
    run_command(
        [cmake, "--build", "build", "--config", "Release", "--target", "llama-cli", "llama-quantize", "llama-server"],
        log_step="compile",
    )

def main():
    setup_gguf()
    gen_code()
    compile()
    prepare_model()
    
def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(description='Setup the environment for running the inference')
    parser.add_argument("--hf-repo", "-hr", type=str, help="Model used for inference", choices=SUPPORTED_HF_MODELS.keys())
    parser.add_argument("--model-dir", "-md", type=str, help="Directory to save/load the model", default="models")
    parser.add_argument("--log-dir", "-ld", type=str, help="Directory to save the logging info", default="logs")
    parser.add_argument("--quant-type", "-q", type=str, help="Quantization type", choices=SUPPORTED_QUANT_TYPES[arch], default="i2_s")
    parser.add_argument("--quant-embd", action="store_true", help="Quantize the embeddings to f16")
    parser.add_argument("--use-pretuned", "-p", action="store_true", help="Use the pretuned kernel parameters")
    return parser.parse_args()

def signal_handler(sig, frame):
    logging.info("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    main()
