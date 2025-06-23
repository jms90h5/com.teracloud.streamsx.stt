# Makefile for NeMo CTC Speech-to-Text Sample
# Based on Streams sample pattern

.PHONY: build all clean

STREAMS_INSTALL ?= $(HOME)/teracloud/streams/7.2.0.0
TOOLKIT_NAME = com.teracloud.streamsx.stt
TOOLKIT_PATH = ..

SPL_MAIN_COMPOSITE = com.teracloud.streamsx.stt.sample::BasicNeMoTest

SPLC = $(STREAMS_INSTALL)/bin/sc
DATA_DIR = data
OUTPUT_DIR = output

all: build

build:
	@echo "Building BasicNeMoTest sample..."
	@mkdir -p $(OUTPUT_DIR)
	$(SPLC) -a -t $(TOOLKIT_PATH) -M $(SPL_MAIN_COMPOSITE) --output-directory $(OUTPUT_DIR)
	@echo "✅ BasicNeMoTest built successfully"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OUTPUT_DIR)
	@echo "✅ Clean completed"

submit: build
	@echo "Submitting BasicNeMoTest job..."
	streamtool submitjob $(OUTPUT_DIR)/com.teracloud.streamsx.stt.sample.BasicNeMoTest.sab
	@echo "✅ Job submitted"

test: submit
	@echo "Checking job status..."
	streamtool lsjobs

status:
	@echo "Toolkit: $(TOOLKIT_PATH)"
	@echo "Main Composite: $(SPL_MAIN_COMPOSITE)"
	@echo "Output: $(OUTPUT_DIR)"
	@ls -la $(TOOLKIT_PATH)/impl/lib/libnemo_ctc_impl.so 2>/dev/null && echo "✅ Implementation library found" || echo "❌ Implementation library missing"
	@ls -la $(TOOLKIT_PATH)/models/fastconformer_ctc_export/model.onnx 2>/dev/null && echo "✅ ONNX model found" || echo "❌ ONNX model missing"