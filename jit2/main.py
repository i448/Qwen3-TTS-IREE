import iree.turbine.aot as aot
from compile import fxb

def main():
    print("Exporting Tokenizer V2 (Encoder & Decoder) to MLIR...")
    
    # Get the ExportOutput from FxProgramsBuilder
    # This captures the programs added via @fxb.export_program
    try:
        exported_module = fxb.export()
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # print(exported_module.mlir_module) # For debug
    
    # To compile to VMFB:
    # binary = exported_module.compile(target_backends=["llvm-cpu"])
    # with open("tokenizer_v2.vmfb", "wb") as f:
    #     f.write(binary)
    # print("Compiled to tokenizer_v2.vmfb")

    print("\nSuccessfully exposed:")
    print(" - encode(input_values, padding_mask) -> codes")
    print(" - decode(codes) -> audio_values")
    print(" - speaker_forward(mels) -> speaker_embedding")
    print(" - talker_forward(inputs_embeds, attention_mask) -> hidden_states/logits")
    print("\nReady for compilation via fxb.export().compile(...)")

if __name__ == "__main__":
    main()
